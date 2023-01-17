!pip install torch --upgrade
import torch

import time

import numpy as np

import pandas as pd

from numba import njit



# Reproducibility

torch.manual_seed(20191210)



# Constants

N_DAYS = 100

N_FAMILIES = 5000

MAX_OCCUPANCY = 300

MIN_OCCUPANCY = 125

INPUT_PATH = '/kaggle/input/santa-workshop-tour-2019/'

OUTPUT_PATH = ''



# The code will run on `cuda`, however there is a bug that I have not chased down and it will not converge

# The speed boost from GPU is not that great, either - vectorising the PyTorch score method could make it competitive

DEFAULT_DEVICE = torch.device('cpu')

# Agent hyperparams

MAX_ACTION = 4

SOFT_PENALTY_PER_PERSON = 1000

PENALTY_RAMP_TIME = 2000  # Number of batches over which to ramp up the soft penalty and accounting costs

BATCH_SIZE = 1000

N_BATCHES = 6000



# Optimiser hyperparams

LR = 0.025

GRADIENT_CLIP = 100.0

MAX_PREFERENCE = 8.5

USE_ADAM = True



#  Only used if USE_ADAM=True

ADAM_BETA_M = 0.9

ADAM_BETA_V = 0.99

ADAM_EPSILON = 0.000001



#  Only used if USE_ADAM=False

MOMENTUM = 0.95
@njit

def faster_cost(allocation, cost_matrix, family_size, days, account_rate, soft_penalty):

    penalty = 0

    daily_occupancy = np.zeros(N_DAYS+1)

    for i in range(N_FAMILIES):

        n = family_size[i]

        d = allocation[i]

        daily_occupancy[d] += n

        penalty += cost_matrix[i, d]



    relevant_occupancy = daily_occupancy[1:]



    for day in days:

        today_count = daily_occupancy[day]

        if today_count > MAX_OCCUPANCY:

            penalty += soft_penalty * (today_count - MAX_OCCUPANCY)

            daily_occupancy[day] = MAX_OCCUPANCY

        elif today_count < MIN_OCCUPANCY:

            penalty += soft_penalty * (MIN_OCCUPANCY - today_count)

            daily_occupancy[day] = MIN_OCCUPANCY



    init_occupancy = daily_occupancy[days[0]]

    accounting_cost = (init_occupancy - 125.0) / 400.0 * init_occupancy**(0.5)

    accounting_cost = max(0, accounting_cost)



    yesterday_count = init_occupancy

    for day in days[1:]:

        today_count = daily_occupancy[day]

        diff = np.abs(today_count - yesterday_count) * 0.02 * account_rate

        accounting_cost += max(0, (today_count - 125.0) / 400.0 * today_count**(0.5 + diff))

        yesterday_count = today_count



    penalty += account_rate * accounting_cost



    return penalty

class Problem:

    def __init__(self, base_path, device):

        self.base_path = base_path

        self.data = pd.read_csv(base_path + 'family_data.csv', index_col='family_id')

        self.device = device

        self._build()

        self._set_buffers()

        self.days = np.arange(N_DAYS, 0, -1)

        self.np_costs_matrix = self.costs_matrix.cpu().numpy()

        self.np_family_size = self.family_size.cpu().numpy()



    def _build(self):

        self.family_size = torch.tensor(self.data.n_people.values, dtype = torch.int, device = self.device)

        self.penalties_array = torch.tensor([

            [

                0,

                50,

                50 + 9 * n,

                100 + 9 * n,

                200 + 9 * n,

                200 + 18 * n,

                300 + 18 * n,

                300 + 36 * n,

                400 + 36 * n,

                500 + 36 * n + 199 * n,

                500 + 36 * n + 398 * n

            ]

            for n in range(self.family_size.max() + 1)

        ], device = self.device)

        self.actions_to_days = torch.full((self.data.shape[0], 10), 1, dtype = torch.long, device = self.device)

        self.costs_matrix = torch.full((self.data.shape[0], N_DAYS + 1), -1, device = self.device)



        for i, choice in enumerate(self.data.loc[:, 'choice_0': 'choice_9'].values):

            n = self.family_size[i]

            for day in range(1,101):

                self.costs_matrix[i, day] = self.penalties_array[n,-1]

            for d, day in enumerate(choice):

                self.costs_matrix[i, day] = self.penalties_array[n, d]

                self.actions_to_days[i, d] = int(day)



    def _set_buffers(self):

        self.daily_occupancy = torch.zeros(N_DAYS+2, dtype = torch.int, device = self.device)

        self.penalties = torch.zeros(N_FAMILIES, device = self.device)

        self.excess = torch.zeros(N_DAYS+2, dtype = torch.int, device = self.device)

        self.shortage = torch.zeros(N_DAYS+2, dtype = torch.int, device = self.device)

        self.diff_counts = torch.zeros(N_DAYS, dtype = torch.int, device = self.device)

        self.today_cost_factor = torch.zeros(N_DAYS, dtype = torch.int, device = self.device)

        self.today_cost_factor_f = torch.zeros(N_DAYS, dtype = torch.float, device = self.device)

        self.diff_counts_factor_f = torch.zeros(N_DAYS, dtype = torch.float, device = self.device)

        self.day_power_f = torch.zeros(N_DAYS, dtype = torch.float, device = self.device)

        self.day_costs = torch.zeros(N_DAYS, dtype = torch.float, device = self.device)



    def cost_function(self, assignments, account_rate = 1.0, soft_penalty = SOFT_PENALTY_PER_PERSON):

        # Calculate days

        self.daily_occupancy.zero_()

        self.daily_occupancy.index_add_(0, assignments, self.family_size)

        self.daily_occupancy[0] = MIN_OCCUPANCY

        self.daily_occupancy[-1] = MIN_OCCUPANCY



        # Individual family costs

        torch.gather(self.costs_matrix, 1, assignments.view(-1,1), out=self.penalties)

        penalty = self.penalties.sum().item()



        # Soft penalty

        torch.clamp(self.daily_occupancy, MAX_OCCUPANCY, 10000000, out=self.excess)

        self.excess.add_(-MAX_OCCUPANCY)

        torch.clamp(self.daily_occupancy, 0, MIN_OCCUPANCY, out=self.shortage)

        self.shortage.add_(-MIN_OCCUPANCY)



        # Clamped to prevent nonlinear penalties dominating soft penalties

        self.daily_occupancy.clamp_(MIN_OCCUPANCY, MAX_OCCUPANCY)



        penalty += (self.excess.sum() - self.shortage.sum()).item() * soft_penalty



        # Accounting costs

        today_counts = self.daily_occupancy.narrow(0, 1, N_DAYS)

        yesterday_counts = self.daily_occupancy.narrow(0, 2, N_DAYS)

        torch.neg(yesterday_counts, out=self.diff_counts)

        # All this is so that we use the buffers throughout and avoid spawning new tensors for interim calculations

        self.diff_counts.add_(today_counts)

        self.diff_counts.abs_()

        self.diff_counts[N_DAYS-1] = 0

        torch.add(today_counts, -MIN_OCCUPANCY, out = self.today_cost_factor)

        torch.mul(self.today_cost_factor, 0.0025, out = self.today_cost_factor_f)

        torch.mul(self.diff_counts, 0.02 * account_rate, out = self.diff_counts_factor_f)

        self.diff_counts_factor_f.add_(0.5)



        # TODO: Check whether today_counts.type(torch.float) is a view or needs a buffer and how?

        torch.pow(today_counts.type(torch.float), self.diff_counts_factor_f, out=self.day_power_f)

        self.today_cost_factor_f.mul_(self.day_power_f)



        accounting_cost = self.today_cost_factor_f.sum().item()



        penalty += account_rate * accounting_cost



        return penalty



    def fast_cost_function(self, assignments, account_rate = 1.0, soft_penalty = SOFT_PENALTY_PER_PERSON):

        return faster_cost(assignments.numpy(), self.np_costs_matrix, self.np_family_size, self.days, account_rate, soft_penalty)



    def is_valid(self, assignments):

        # Calculate days

        self.daily_occupancy.zero_()

        self.daily_occupancy.index_add_(0, assignments, self.family_size)

        self.daily_occupancy[0] = MIN_OCCUPANCY

        self.daily_occupancy[-1] = MIN_OCCUPANCY



        torch.clamp(self.daily_occupancy, MAX_OCCUPANCY, 10000000, out=self.excess)

        self.excess.add_(-MAX_OCCUPANCY)

        torch.clamp(self.daily_occupancy, 0, MIN_OCCUPANCY, out=self.shortage)

        self.shortage.add_(-MIN_OCCUPANCY)



        mismatch = (self.excess.sum() - self.shortage.sum()).item()



        return (mismatch == 0)

class Submission:

    def __init__(self, sample_path, solutions_path):

        self.sample_path = sample_path

        self.solutions_path = solutions_path

        self.load_sample_solution()

        self.valid_write_point = 85000.0



    def load_sample_solution(self):

        # This is just so we have a template to write with

        self.table = pd.read_csv(self.sample_path + 'sample_submission.csv', index_col='family_id')



    def write(self, torch_data, score):

        self.table['assigned_day'] = torch_data.cpu().numpy()

        path = f'{self.solutions_path}submission_{int(score)}.csv'

        self.table.to_csv(path)

        print(f'Wrote {path}')



    def get_assigned_days(self):

        return self.table['assigned_day'].values



    def update_write_point(self, actual_score):

        if self.valid_write_point > 80000:

            self.valid_write_point = min(actual_score - 0.1, self.valid_write_point - 1000)

            return



        if self.valid_write_point > 75000:

            self.valid_write_point = min(actual_score - 0.1, self.valid_write_point - 500)

            return



        if self.valid_write_point > 70000:

            self.valid_write_point = min(actual_score - 0.1, self.valid_write_point - 100)

            return



        if self.valid_write_point > 69000:

            self.valid_write_point = min(actual_score - 0.1, self.valid_write_point - 20)

            return



        self.valid_write_point = actual_score - 0.1



    def write_if_better(self, torch_data, score):

        if score >= self.valid_write_point:

            return



        self.write(torch_data, score)



        self.update_write_point(score)
class Optimiser:

    def __init__(self, device):

        self.device = device

        if USE_ADAM:

            self.adam_beta_m_product = ADAM_BETA_M

            self.adam_beta_v_product = ADAM_BETA_V

            self.adam_m = torch.zeros(N_FAMILIES, MAX_ACTION, device = self.device)

            self.adam_v = torch.zeros(N_FAMILIES, MAX_ACTION, device = self.device)

        else:

            self.gradient_vs = torch.zeros(N_FAMILIES, MAX_ACTION, device = self.device)



    def update(self, preferences, gradients):

        if USE_ADAM:

            self.adam_m = ADAM_BETA_M * self.adam_m + (1 - ADAM_BETA_M) * gradients

            self.adam_v = ADAM_BETA_V * self.adam_v + (1 - ADAM_BETA_V) * gradients * gradients

            m_hat = self.adam_m/(1 - self.adam_beta_m_product)

            v_hat = self.adam_v/(1 - self.adam_beta_v_product)

            adam_update = m_hat / (torch.sqrt(v_hat) + ADAM_EPSILON)

            self.adam_beta_m_product *= ADAM_BETA_M

            self.adam_beta_v_product *= ADAM_BETA_V



            # Gradient *descent* because we have a cost to minimise, not a reward to maximise

            preferences -= LR * adam_update

        else:

            # Simple momentum-based update

            self.gradient_vs = (MOMENTUM * self.gradient_vs) + gradients

            preferences -= LR * gradient_vs



        preferences = (preferences - preferences.max(axis=1).values.view(-1,1)) + MAX_PREFERENCE

        preferences.clamp_(0.0, MAX_PREFERENCE)



        return preferences

class Agent:

    def __init__(self, problem, optimiser, device):

        self.device = device

        self.problem = problem

        self.optimiser = optimiser



        # Starting preferences

        self.preferences = torch.zeros(N_FAMILIES, MAX_ACTION, device = self.device)



        # Buffers

        self.w = torch.zeros(N_FAMILIES, MAX_ACTION, device = self.device)

        self.z = torch.zeros(N_FAMILIES, 1, device = self.device)

        self.actions = torch.zeros(N_FAMILIES, BATCH_SIZE, dtype=torch.long, device = self.device)

        self.days = torch.zeros(N_FAMILIES, BATCH_SIZE, dtype=torch.long, device = self.device)



        self.best_score = 100000000.0

        self.best_item = None

        self.ramp_speed = 1.0/PENALTY_RAMP_TIME



    def _get_batch(self):

        torch.exp(self.preferences, out=self.w)

        torch.sum(self.w, axis=1, keepdims=True, out=self.z)

        self.w.div_(self.z)

        torch.multinomial(self.w, BATCH_SIZE, replacement=True, out=self.actions)

        torch.gather(self.problem.actions_to_days, 1, self.actions, out=self.days)

        return (self.w, self.actions.T, self.days.T)



    def _get_greedy(self):

        actions = self.preferences.argmax(axis=1).view(-1,1)

        days = self.problem.actions_to_days.gather(1, actions)

        return days.T[0]



    def _get_batch_scores(self, days_allocated, batch_id):

        batch_scores = []

        sp = min(batch_id * self.ramp_speed * SOFT_PENALTY_PER_PERSON, SOFT_PENALTY_PER_PERSON)

        ar = min(1.0, batch_id * self.ramp_speed)



        for i in range(BATCH_SIZE):

            assignments = days_allocated[i]

            score = self.problem.fast_cost_function(assignments, account_rate = ar, soft_penalty = sp)

            if score < self.best_score:

                self.best_score = score

                self.best_item = assignments.clone().detach()



            batch_scores.append( score )



        return batch_scores



    def _get_batch_gradients(self, probs, actions, batch_scores):

        gradients =  x = torch.zeros(N_FAMILIES, MAX_ACTION, device = self.device)



        for i in range(BATCH_SIZE):

            score = batch_scores[i]

            relative_score = (score - self.baseline)



            # First assume none of actions were taken

            # If we didn't choose an action and the score increased over baseline, then

            # choosing the unused action instead *might* decrease the score

            this_gradient = -relative_score * probs



            # Find the actions actually taken

            selector = actions[i].reshape(-1,1)

            selected_probs = probs.gather(1, selector)



            # With the taken actions - if we chose an item when the score was higher, then

            # making this action more probable should increase the score

            chosen_gradients = relative_score * (1 - selected_probs)

            this_gradient.scatter_(1, selector, chosen_gradients)



            # Accumulate

            gradients += this_gradient



        gradients /= BATCH_SIZE

        gradients.clamp_(-GRADIENT_CLIP, GRADIENT_CLIP)



        return gradients



    def process_batch(self, batch_id):

        t = time.clock_gettime(0)

        probs, actions, days_allocated = self._get_batch()

        batch_scores = self._get_batch_scores(days_allocated, batch_id)



        if batch_id > 0:

            gradients = self._get_batch_gradients(probs, actions, batch_scores)

            self.preferences = self.optimiser.update(self.preferences, gradients)



            # Whilst ramping up, best_score needs adjusting to new realities

            if (batch_id <= PENALTY_RAMP_TIME):

                sp = min(batch_id * SOFT_PENALTY_PER_PERSON * self.ramp_speed, SOFT_PENALTY_PER_PERSON)

                ar = min(1.0, batch_id * self.ramp_speed)

                self.best_score = self.problem.cost_function(self.best_item, account_rate = ar, soft_penalty = sp)



        self.baseline = np.array(batch_scores).mean()



        self.greedy_item = self._get_greedy()

        self.greedy_score = self.problem.cost_function(self.greedy_item)

        t = time.clock_gettime(0) - t



        if batch_id % 100 == 0:

            print("Batch {} ({}s). Mean {}. Best {}. Greedy {}".format(

                   batch_id, round(t,1), int(np.array(batch_scores).mean()),

                   int(self.best_score), int(self.greedy_score) ) )

def main():

    submission = Submission(INPUT_PATH, OUTPUT_PATH)



    problem = Problem(INPUT_PATH, DEFAULT_DEVICE)

    optimiser = Optimiser(DEFAULT_DEVICE)

    agent = Agent(problem, optimiser, DEFAULT_DEVICE)



    for batch_id in range(N_BATCHES):

        agent.process_batch(batch_id)



        if agent.greedy_score < submission.valid_write_point and problem.is_valid(agent.greedy_item):

            submission.write_if_better(agent.greedy_item, agent.greedy_score)



        if batch_id > PENALTY_RAMP_TIME and agent.best_score < submission.valid_write_point and problem.is_valid(agent.best_item):

            submission.write_if_better(agent.best_item, agent.best_score)

            

    submission.valid_write_point = 100000

    

    # A bit fiddly at the end, as we are never sure that the scores represent anything valid

    if agent.greedy_score > agent.best_score:

        if problem.is_valid(agent.best_item):

            submission.write_if_better(agent.best_item, agent.best_score)

        if problem.is_valid(agent.greedy_item):

            submission.write_if_better(agent.greedy_item, agent.greedy_score)

    else:

        if problem.is_valid(agent.greedy_item):

            submission.write_if_better(agent.greedy_item, agent.greedy_score)

        if problem.is_valid(agent.best_item):

            submission.write_if_better(agent.best_item, agent.best_score)

main()



print("DONE")