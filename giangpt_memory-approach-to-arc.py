import numpy as np





class Kernel(object):

    def __init__(self):

        self.memory = None

        self.d, self.k = None, None  # d: distance (num neighbour layers), k: num. cells

        self.remove_idxs = None  # position indices to remove some of the cells from kernel.

        self.conclusion_idx = None  # the cell position index being investigated with its neighbours



    def _get_kernel(self, frame, row, col):

        return frame[row-self.d:row+self.d+1, col-self.d:col+self.d+1]



    def _memory_handler(self, row, col, neighs):

        if self.memory:

            memo = self.memory.learn(row, col)

            neighs = np.array(memo.tolist() + neighs.tolist())

        return neighs



    def get_neighbours(self, input_frame, row, col):

        kernel = self._get_kernel(input_frame, row, col)

        neighs = np.delete(kernel.flatten(), self.remove_idxs)

        neighs = self._memory_handler(row, col, neighs)

        return neighs



    def get_label(self, output_frame, row, col):

        kernel = self._get_kernel(output_frame, row, col)

        label = kernel.flatten()[self.conclusion_idx]

        return label





class KernelD1K9(Kernel):

    """

    Kernel to get neighbouring cells (features).



    This kernel returns 8 nearest neighbours and the cell in the middle (conclusion cell).

    """

    def __init__(self, memory=None):

        super().__init__()

        self.memory = memory

        self.len_memory = self.memory.len_output if self.memory else 0

        self.d = 1

        self.k = 9 + self.len_memory

        self.remove_idxs = []

        self.conclusion_idx = 4

        self.indices = np.arange(self.k).tolist()





class KernelD2K25(Kernel):

    def __init__(self, memory=None):

        super().__init__()

        self.memory = memory

        self.len_memory = self.memory.len_output if self.memory else 0

        self.d = 2

        self.k = 25 + self.len_memory

        self.remove_idxs = []

        self.conclusion_idx = 12

        self.indices = np.arange(self.k).tolist()

import numpy as np

import collections





class Memory(object):

    def __init__(self):

        self.max_len = None

        self.direction = None

        self.input_original = None

        self.len_output = None



    def init_memory(self, input_original):

        self.input_original = input_original



    def _walk_vertical(self):

        rows = self.input_original.shape[0]

        r = reversed(range(rows)) if self.direction[2:] == "tb" else range(rows)

        for i in r:

            yield i



    def _walk_horizontal(self):

        cols = self.input_original.shape[1]

        r = reversed(range(cols)) if self.direction[:2] == "rl" else range(cols)

        for i in r:

            yield i





class MemoryLTVHR(Memory):

    def __init__(self, max_len=5, direction="tblr"):

        super().__init__()

        self.direction = direction

        self.max_len = max_len

        self.len_output = 2 * self.max_len

        self.dq_h = collections.deque(maxlen=self.max_len)

        self.dq_v = collections.deque(maxlen=self.max_len)



    def reset(self):

        self.dq_h = collections.deque(maxlen=self.max_len)

        self.dq_v = collections.deque(maxlen=self.max_len)



    def _get_comparison_value(self, dq):

        comp = dq[-1] if len(dq) > 0 else -1

        return comp



    def learn(self, row_idx, col_idx):

        for i in self._walk_horizontal():

            v = self.input_original[row_idx, i]

            if int(v) != self._get_comparison_value(self.dq_h):

                self.dq_h.append(int(v))

        for i in self._walk_vertical():

            v = self.input_original[i, col_idx]

            if int(v) != self._get_comparison_value(self.dq_v):

                self.dq_v.append(int(v))

        l0 = list(self.dq_h) + [999] * (self.max_len - len(self.dq_h))

        l1 = list(self.dq_v) + [999] * (self.max_len - len(self.dq_v))

        memory = np.array(l0 + l1)

        self.reset()

        return memory





class MemoryLTVR(Memory):

    def __init__(self, max_len=5, direction="tblr"):

        super().__init__()

        self.direction = direction

        self.max_len = max_len

        self.len_output = self.max_len

        self.dq_v = collections.deque(maxlen=self.max_len)



    def reset(self):

        self.dq_v = collections.deque(maxlen=self.max_len)



    def _get_comparison_value(self, dq):

        comp = dq[-1] if len(dq) > 0 else 0

        return comp



    def learn(self, row_idx, col_idx):

        for i in self._walk_vertical():

            v = self.input_original[i, col_idx]

            if int(v) != self._get_comparison_value(self.dq_v):

                self.dq_v.append(int(v))

        l = list(self.dq_v) + [999] * (self.max_len - len(self.dq_v))

        memory = np.array(l)

        self.reset()

        return memory





class MemoryLTVHR2N(Memory):

    """

    Memory with 2 neighbours up and down.

    """

    def __init__(self, max_len=5, direction="tblr"):

        super().__init__()

        self.direction = direction

        self.max_len = max_len

        self.len_output = 6 * self.max_len

        self.neigh_range = ["u", "m", "d"]

        self.diff_range = {"u": -1, "m": 0, "d": 1}

        self.dq_h, self.dq_v = {}, {}

        for r in self.neigh_range:

            self.dq_h[r] = collections.deque(maxlen=self.max_len)

            self.dq_v[r] = collections.deque(maxlen=self.max_len)



    def reset(self):

        for r in self.neigh_range:

            self.dq_h[r] = collections.deque(maxlen=self.max_len)

            self.dq_v[r] = collections.deque(maxlen=self.max_len)



    def _get_comparison_value(self, dq):

        comp = dq[-1] if len(dq) > 0 else -1

        return comp



    def learn(self, row_idx, col_idx):

        vh, vv = {}, {}

        for i in self._walk_horizontal():

            for r in self.neigh_range:

                try:

                    vh[r] = self.input_original[row_idx+self.diff_range[r], i]

                    if int(vh[r]) != self._get_comparison_value(self.dq_h[r]):

                        self.dq_h[r].append(int(vh[r]))

                except:

                    pass

        for i in self._walk_vertical():

            for r in self.neigh_range:

                try:

                    vv[r] = self.input_original[i, col_idx+self.diff_range[r]]

                    if int(vv[r]) != self._get_comparison_value(self.dq_v[r]):

                        self.dq_v[r].append(int(vv[r]))

                except:

                    pass

        l = []

        for r in self.neigh_range:

            l_ = list(self.dq_h[r]) + [999] * (self.max_len - len(self.dq_h[r]))

            l = l + l_

        for r in self.neigh_range:

            l_ = list(self.dq_v[r]) + [999] * (self.max_len - len(self.dq_v[r]))

            l = l + l_

        memory = np.array(l)

        self.reset()

        return memory

from pathlib import Path

import numpy as np

import pandas as pd

import json

import matplotlib.pyplot as plt

from matplotlib import colors

import itertools

import os





class AbductiveReasoner(object):

    """

    Abductive Reasoner. Uses neighbouring cell states for generating

    explanations to predict the output for this competition

    https://www.kaggle.com/c/abstraction-and-reasoning-challenge

    created by it-from-bit - https://www.kaggle.com/everyitfrombit.



    Attributes

    ----------

    observations (dict): stores observations as keys and conclusions as values.

    explanations (dict): using observations, creates explanations and stores as keys

    along with the conclusions as values.



    """

    def __init__(self, kernel, config):

        """

        Abductive Reasoner.



        Attributes

        ----------

        data_path (str): Path to the dataset.

        kernel (class): Kernel instance for nearest neighbours (cell features).

        """

        self.cmap = colors.ListedColormap(

            ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',

             '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

        self.norm = colors.Normalize(vmin=0, vmax=9)

        self.task = None

        self.observations = None

        self.explanations = None

        self.color_explanations = None

        self.input_original = None

        self.input_ = None

        self.output = None

        self.config = config

        self.kernel = kernel

        self.data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')

        self._load_tasks()



    def _load_tasks(self):

        self.train_path, self.valid_path = self.data_path / "training", self.data_path / "evaluation"

        self.test_path = self.data_path / "test"

        self.training_tasks = sorted(os.listdir(self.train_path))

        self.evaluation_tasks = sorted(os.listdir(self.valid_path))

        self.test_tasks = sorted(os.listdir(self.test_path))



    def init_task(self, file, is_test=False):

        task_file = None

        if file in self.training_tasks:

            task_file = str(self.train_path / file)

        elif file in self.evaluation_tasks:

            task_file = str(self.valid_path / file)

        elif file in self.test_tasks:

            task_file = str(self.test_path / file)

        with open(task_file, 'r') as f:   

            self.task = json.load(f)

        self.observations, self.explanations = None, None

        self.input_, self.output = None, None

    def isValidTask(self):

        num_train_pairs = len(self.task["train"])

        for task_num in range(num_train_pairs):

            input_color = self.task["train"][task_num]['input']

            target_color = self.task["train"][task_num]['output']

            nrows, ncols = len(input_color), len(input_color[0])

            target_rows, target_cols = len(target_color), len(target_color[0])

            if (target_rows != nrows) or (target_cols != ncols):

                return False

        return True



    def _pad_image(self, image):

        return np.pad(image, self.kernel.d, constant_values=0)



    def _sample_handler(self, sample):

        self.input_ = np.array(sample["input"])

        self.input_ = self._pad_image(self.input_)

        if "output" in sample:

            self.output = np.array(sample["output"])

            self.output = self._pad_image(self.output)

        else:

            self.output = None

        self.input_original = self.input_.copy()

        if self.kernel.memory:

            self.kernel.memory.init_memory(self.input_original)



    def _grid_walk(self, direction):

        rows, cols = self.input_.shape[0], self.input_.shape[1]

        r0 = reversed(range(self.kernel.d, rows-self.kernel.d)) if direction[:2] == "bt" else range(self.kernel.d, rows-self.kernel.d)

        for i in r0:

            r1 = reversed(range(self.kernel.d, cols - self.kernel.d)) if direction[2:] == "rl" else range(self.kernel.d,cols - self.kernel.d)

            for j in r1:

                yield i, j



    def _generate_observation(self, neighs, conclusion=None):

        return neighs.tolist() + [conclusion] if conclusion is not None else neighs.tolist()



    def observe(self):

        num_loops = self.config.get("num_observation_loops", 10)

        walk_directions = self.config.get("observation_walk_directions", ["tblr"])

        train = self.task["train"].copy()

        self.observations = []

        for d in walk_directions:

            for sample in train:

                self._sample_handler(sample)

                for loop in range(num_loops):

                    for i, j in self._grid_walk(d):

                        neighs = self.kernel.get_neighbours(self.input_, i, j)

                        conclusion = self.kernel.get_label(self.output, i, j)

                        if self._sum_neighs(neighs) > 0:

                            observation = self._generate_observation(neighs, conclusion)

                            if observation not in self.observations:

                                self.observations.append(observation)

                            self.input_[i, j] = self.output[i, j]

        self.input_ = self.input_original.copy()  # reset input



    def create_observation_df(self, is_sorted=False):

        df = pd.DataFrame(

            np.array(self.observations),

            columns=[f"feature_{i}" for i in range(len(self.observations[0][:-1]))] + ["conclusion"])

        df = df.sort_values(list(df.columns)) if is_sorted else df

        return df



    def _combination_walk(self):

        r_min = self.config.get("combination_r_min", 1)

        r_max = self.config.get("combination_r_max", self.kernel.k)

        for r in range(r_min, r_max+1):

            for combi in itertools.combinations(self.kernel.indices, r):

                yield combi



    def _explanation_handler(self, explanations, freq_threshold):

        explanations_ = explanations.copy()

        for explanation in explanations.keys():

            if len(set(explanations[explanation]["conclusion"])) == 1:  # no contradiction condition

                freq = len(explanations_[explanation]["conclusion"])

                if freq >= freq_threshold:

                    explanations_[explanation]["frequency"] = freq

                    explanations_[explanation]["conclusion"] = int(explanations[explanation]["conclusion"][0])

                else:

                    del explanations_[explanation]

            else:

                del explanations_[explanation]

        return explanations_



    def _generate_explanation(self, observation, combi):

        explanation = ",".join(

            [str(s) if i in combi else "-" for i, s in enumerate(observation[:self.kernel.k])])

        return explanation



    def reason(self):

        freq_threshold = self.config.get("frequency_threshold", 2)

        explanations = {}

        for combi in self._combination_walk():

            for observation in self.observations:

                explanation = self._generate_explanation(observation, combi)

                if explanation in explanations:

                    explanations[explanation]["conclusion"].append(observation[-1])

                else:

                    explanations[explanation] = {"conclusion": [observation[-1]]}

        self.explanations = self._explanation_handler(explanations, freq_threshold)



    def create_explanation_df(self, is_sorted=False, is_color=False):

        explanations = self.color_explanations if is_color else self.explanations

        rows = []

        for explanation in explanations.keys():

            con = explanations[explanation]["conclusion"]

            freq = explanations[explanation]["frequency"]

            rows.append([s for s in explanation.split(",")] + [con] + [freq])

        columns = [f"feature_{i}" for i in range(self.kernel.k)] + ["conclusion"] + ["frequency"]

        df = pd.DataFrame(np.array(rows), columns=columns)

        df = df.sort_values(list(df.columns)) if is_sorted else df

        return df



    def _observation_encoder(self, observation):

        d = list(dict.fromkeys(observation))

        encoded_observation = [str(d.index(c)) for c in observation]

        return encoded_observation, d



    def explain_color(self):

        freq_threshold = self.config.get("color_frequency_threshold", 2)

        explanations = {}

        for combi in self._combination_walk():

            for observation in self.observations:

                observation, _ = self._observation_encoder(observation)

                explanation = self._generate_explanation(observation, combi)

                if explanation in explanations:

                    explanations[explanation]["conclusion"].append(observation[-1])

                else:

                    explanations[explanation] = {"conclusion": [observation[-1]]}

        self.color_explanations = self._explanation_handler(explanations, freq_threshold)



    def _decide_conclusion(self, conclusions):

        conclusion = None

        val = - np.inf

        df = pd.DataFrame(conclusions, columns=["conclusion", "frequency", "level"])

        for conc in df.conclusion.unique():

            val_ = df[(df.conclusion == conc) & (df.level == "color_level")].frequency.shape[0]

            if not val_:

                val_ = df[(df.conclusion == conc) & (df.level == "first_level")].frequency.shape[0]

            if val_ > val:

                conclusion, val = conc, val_

        return conclusion



    def _sum_neighs(self, neighs):

        len_memory = self.kernel.len_memory

        return neighs[len_memory if len_memory != 0 else None:].sum()



    def _remove_padding(self, frame):

        return frame[self.kernel.d: -self.kernel.d, self.kernel.d: -self.kernel.d]



    def _revert_sample_padding(self):

        self.input_original = self._remove_padding(self.input_original)

        self.input_ = self._remove_padding(self.input_)

        if self.output is not None:

            self.output = self._remove_padding(self.output)



    def _compute_score(self, prediction):

        score = 0

        if self.output is not None:

            self._revert_sample_padding()

            score = 1 if np.array_equal(self.output, prediction) else 0

        return score



    def predict(self, is_train=False, visualize=False):

        num_loops = self.config.get("num_inference_loops", 10)

        visualize_prediction = self.config.get("visualize_prediction", False)

        walk_directions = self.config.get("prediction_walk_directions", ["tblr"])

        samples = self.task["test"] if not is_train else self.task["train"]

        predictions, scores = [], []

        for d in walk_directions:

            for sample in samples:

                self._sample_handler(sample)

                prediction = self.input_.copy()

                for loop in range(num_loops):

                    for i, j in self._grid_walk(d):

                        neighs = self.kernel.get_neighbours(prediction, i, j)

                        if self._sum_neighs(neighs) > 0:

                            explanation_set, conclusions = [], []

                            for combi in self._combination_walk():

                                observation = self._generate_observation(neighs)

                                explanation = self._generate_explanation(observation, combi)

                                encoded_obs, color_set = self._observation_encoder(observation)

                                encoded_explanation = self._generate_explanation(encoded_obs, combi)

                                try:

                                    con = self.color_explanations[encoded_explanation]["conclusion"]

                                    freq = self.color_explanations[encoded_explanation]["frequency"]

                                    con = color_set[con]

                                    conclusions.append((con, freq, "color_level"))

                                    explanation_set.append(encoded_explanation)

                                except:

                                    if explanation in self.explanations:

                                        con = self.explanations[explanation]["conclusion"]

                                        freq = self.explanations[explanation]["frequency"]

                                        conclusions.append((con, freq, "first_level"))

                                        explanation_set.append(explanation)

                            conclusion = self._decide_conclusion(conclusions)

                            prediction[i, j] = conclusion if conclusion is not None else prediction[i, j]

                            if visualize_prediction:

                                self.plot_sample(prediction)

                prediction = self._remove_padding(prediction)

                predictions.append(prediction)

                scores.append(self._compute_score(prediction))

                if visualize:

                    self.plot_sample(prediction)

        return predictions, scores



    def save_prediction(self, prediction):

        self.plot_sample(prediction)



    def plot_pictures(self, pictures, labels):

        fig, axs = plt.subplots(1, len(pictures), figsize=(2 * len(pictures), 32))

        for i, (pict, label) in enumerate(zip(pictures, labels)):

            axs[i].imshow(np.array(pict), cmap=self.cmap, norm=self.norm)

            axs[i].set_title(label)

        plt.show()



    def save_pictures(self, pictures, labels, path):

        fig, axs = plt.subplots(1, len(pictures), figsize=(2 * len(pictures), 2 * len(pictures)))

        for i, (pict, label) in enumerate(zip(pictures, labels)):

            axs[i].imshow(np.array(pict), cmap=self.cmap, norm=self.norm)

            axs[i].set_title(label)

        fig.savefig(path)

        plt.close()



    def plot_sample(self, predict=None):

        pictures = [self.input_original, self.output] if self.output is not None else [self.input_original]

        labels = ['Input', 'Output'] if self.output is not None else ["Input"]

        if predict is not None:

            pictures = pictures + [predict]

            labels = labels + ["Predict"]

        self.plot_pictures(pictures, labels)



    def save_samples(self, path, predictions=None, is_train=False):

        samples = self.task["test"] if not is_train else self.task["train"]

        predictions = [None] * len(samples) if predictions is None else predictions

        for i, (prediction, sample) in enumerate(zip(predictions, samples)):

            self._sample_handler(sample)

            self._revert_sample_padding()

            pictures = [self.input_original, self.output] if self.output is not None else [self.input_original]

            labels = ['Input', 'Output'] if self.output is not None else ["Input"]

            if prediction is not None:

                pictures = pictures + [prediction]

                labels = labels + ["Predict"]

            path_ = Path(path).parent / f"{Path(path).stem}_{i}{Path(path).suffix}"

            self.save_pictures(pictures, labels, path_)



    def plot_train(self):

        train = self.task["train"]

        for sample in train:

            self._sample_handler(sample)

            self.plot_sample()

config = {

        "dataset": "/kaggle/input/abstraction-and-reasoning-challenge/",

        "observation_walk_directions": ["btlr"],

        "prediction_walk_directions": ["btlr"],

        "color_frequency_threshold": 2,

        "frequency_threshold": 2,

        "combination_r_min": 1,

        "combination_r_max": 3,

        "num_observation_loops": 1,

        "num_inference_loops": 1,

        "visualize_prediction": False

    }
file_name = "db3e9e38.json"

memory = MemoryLTVHR(max_len=6)

kernel = KernelD1K9(memory)  # kernel to get nearest neighbours and the cell in question as the features.



reasoner = AbductiveReasoner(kernel, config)

reasoner.init_task(file_name)

reasoner.observe()

reasoner.reason()

reasoner.explain_color()

prediction, scores = reasoner.predict(visualize=True)
data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')

training_path = data_path / 'training'

evaluation_path = data_path / 'evaluation'

test_path = data_path / 'test'
def solutionBox(file_name):

    memory = MemoryLTVHR(max_len=6)

    kernel = KernelD1K9(memory)  # kernel to get nearest neighbours and the cell in question as the features.



    reasoner = AbductiveReasoner(kernel, config)

    reasoner.init_task(file_name)

    if (not reasoner.isValidTask()):

        return None, 0

    reasoner.observe()

    reasoner.reason()

    reasoner.explain_color()

    prediction, scores = reasoner.predict(visualize=False)

    return prediction, scores

    
sample_sub1 = pd.read_csv(data_path/'sample_submission.csv')

sample_sub1 = sample_sub1.set_index('output_id')

sample_sub1.head()
def flattener(pred):

    str_pred = str([row for row in pred])

    str_pred = str_pred.replace(', ', '')

    str_pred = str_pred.replace('[[', '|')

    str_pred = str_pred.replace('][', '|')

    str_pred = str_pred.replace(']]', '|')

    return str_pred
# mode = 'train'

mode = 'test'

if mode=='eval':

    task_path = evaluation_path

elif mode=='train':

    task_path = training_path

elif mode=='test':

    task_path = test_path

    

overall_score = 0



all_task_ids = sorted(os.listdir(task_path))



for task_id in all_task_ids:

    predictions, scores = solutionBox(task_id)

    if predictions == None:

        continue

    if np.array(scores).sum() == len(scores):

        overall_score += 1

    for task_num in range(len(predictions)):

        preds = predictions[task_num]

        preds = preds.astype(int).tolist()

        sample_sub1.loc[f'{task_id[:-5]}_{task_num}',

                       'output']  = flattener(preds)

        

print(overall_score)

    
sample_sub1.head()

sample_sub1.to_csv('submission1.csv')
data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')

training_path = data_path / 'training'

evaluation_path = data_path / 'evaluation'

test_path = data_path / 'test'

training_tasks = sorted(os.listdir(training_path))

eval_tasks = sorted(os.listdir(evaluation_path))





T = training_tasks

Trains = []

for i in range(400):

    task_file = str(training_path / T[i])

    task = json.load(open(task_file, 'r'))

    Trains.append(task)

    

E = eval_tasks

Evals= []

for i in range(400):

    task_file = str(evaluation_path / E[i])

    task = json.load(open(task_file, 'r'))

    Evals.append(task)

    

    

cmap = colors.ListedColormap(

    ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',

     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

norm = colors.Normalize(vmin=0, vmax=9)

# 0:black, 1:blue, 2:red, 3:greed, 4:yellow,

# 5:gray, 6:magenta, 7:orange, 8:sky, 9:brown

plt.figure(figsize=(5, 2), dpi=200)

plt.imshow([list(range(10))], cmap=cmap, norm=norm)

plt.xticks(list(range(10)))

plt.yticks([])

plt.show()



def plot_task(task):

    n = len(task["train"]) + len(task["test"])

    fig, axs = plt.subplots(2, n, figsize=(4*n,8), dpi=50)

    plt.subplots_adjust(wspace=0, hspace=0)

    fig_num = 0

    for i, t in enumerate(task["train"]):

        t_in, t_out = np.array(t["input"]), np.array(t["output"])

        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)

        axs[0][fig_num].set_title(f'Train-{i} in')

        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))

        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))

        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)

        axs[1][fig_num].set_title(f'Train-{i} out')

        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))

        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))

        fig_num += 1

    for i, t in enumerate(task["test"]):

        t_in, t_out = np.array(t["input"]), np.array(t["output"])

        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)

        axs[0][fig_num].set_title(f'Test-{i} in')

        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))

        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))

        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)

        axs[1][fig_num].set_title(f'Test-{i} out')

        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))

        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))

        fig_num += 1

    

    plt.tight_layout()

    plt.show()

    



def plot_picture(x):

    plt.imshow(np.array(x), cmap = cmap, norm = norm)

    plt.show()

    

    

def Defensive_Copy(A): 

    n = len(A)

    k = len(A[0])

    L = np.zeros((n,k), dtype = int)

    for i in range(n):

        for j in range(k):

            L[i,j] = 0 + A[i][j]

    return L.tolist()





def Create(task, task_id = 0):

    n = len(task['train'])

    Input = [Defensive_Copy(task['train'][i]['input']) for i in range(n)]

    Output = [Defensive_Copy(task['train'][i]['output']) for i in range(n)]

    Input.append(Defensive_Copy(task['test'][task_id]['input']))

    return Input, Output





def Recolor(task):

    Input = task[0]

    Output = task[1]

    Test_Picture = Input[-1]

    Input = Input[:-1]

    N = len(Input)

    

    for x, y in zip(Input, Output):

        if len(x) != len(y) or len(x[0]) != len(y[0]):

            return -1

        

    Best_Dict = -1

    Best_Q1 = -1

    Best_Q2 = -1

    Best_v = -1

    # v ranges from 0 to 3. This gives an extra flexibility of measuring distance from any of the 4 corners

    Pairs = []

    for t in range(15):

        for Q1 in range(1,8):

            for Q2 in range(1,8):

                if Q1+Q2 == t:

                    Pairs.append((Q1,Q2))

                    

    for Q1, Q2 in Pairs:

        for v in range(4):

    

  

            if Best_Dict != -1:

                continue

            possible = True

            Dict = {}

                      

            for x, y in zip(Input, Output):

                n = len(x)

                k = len(x[0])

                for i in range(n):

                    for j in range(k):

                        if v == 0 or v ==2:

                            p1 = i%Q1

                        else:

                            p1 = (n-1-i)%Q1

                        if v == 0 or v ==3:

                            p2 = j%Q2

                        else :

                            p2 = (k-1-j)%Q2

                        color1 = x[i][j]

                        color2 = y[i][j]

                        if color1 != color2:

                            rule = (p1, p2, color1)

                            if rule not in Dict:

                                Dict[rule] = color2

                            elif Dict[rule] != color2:

                                possible = False

            if possible:

                

                # Let's see if we actually solve the problem

                for x, y in zip(Input, Output):

                    n = len(x)

                    k = len(x[0])

                    for i in range(n):

                        for j in range(k):

                            if v == 0 or v ==2:

                                p1 = i%Q1

                            else:

                                p1 = (n-1-i)%Q1

                            if v == 0 or v ==3:

                                p2 = j%Q2

                            else :

                                p2 = (k-1-j)%Q2

                           

                            color1 = x[i][j]

                            rule = (p1,p2,color1)

                            

                            if rule in Dict:

                                color2 = 0 + Dict[rule]

                            else:

                                color2 = 0 + y[i][j]

                            if color2 != y[i][j]:

                                possible = False 

                if possible:

                    Best_Dict = Dict

                    Best_Q1 = Q1

                    Best_Q2 = Q2

                    Best_v = v

                

                

    if Best_Dict == -1:

        return -1 #meaning that we didn't find a rule that works for the traning cases

    

    #Otherwise there is a rule: so let's use it:

    n = len(Test_Picture)

    k = len(Test_Picture[0])

    

    answer = np.zeros((n,k), dtype = int)

   

    for i in range(n):

        for j in range(k):

            if Best_v == 0 or Best_v ==2:

                p1 = i%Best_Q1

            else:

                p1 = (n-1-i)%Best_Q1

            if Best_v == 0 or Best_v ==3:

                p2 = j%Best_Q2

            else :

                p2 = (k-1-j)%Best_Q2

           

            color1 = Test_Picture[i][j]

            rule = (p1, p2, color1)

            if (p1, p2, color1) in Best_Dict:

                answer[i][j] = 0 + Best_Dict[rule]

            else:

                answer[i][j] = 0 + color1

                                    

           

            

    return answer.tolist()





Function = Recolor



training_examples = []

for i in range(400):

    task = Trains[i]

    basic_task = Create(task,0)

    a = Function(basic_task)

  

    if  a != -1 and task['test'][0]['output'] == a:

        plot_picture(a)

        plot_task(task)

        print(i)

        training_examples.append(i)

        

        

evaluation_examples = []





for i in range(400):

    task = Evals[i]

    basic_task = Create(task,0)

    a = Function(basic_task)

    

    if a != -1 and task['test'][0]['output'] == a:

       

        plot_picture(a)

        plot_task(task)

        print(i)

        evaluation_examples.append(i)

        

        

sample_sub2 = pd.read_csv(data_path/ 'sample_submission.csv')

sample_sub2.head()





def flattener(pred):

    str_pred = str([row for row in pred])

    str_pred = str_pred.replace(', ', '')

    str_pred = str_pred.replace('[[', '|')

    str_pred = str_pred.replace('][', '|')

    str_pred = str_pred.replace(']]', '|')

    return str_pred





example_grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

display(example_grid)

print(flattener(example_grid))



Solved = []

Problems = sample_sub2['output_id'].values

Proposed_Answers = []

for i in  range(len(Problems)):

    output_id = Problems[i]

    task_id = output_id.split('_')[0]

    pair_id = int(output_id.split('_')[1])

    f = str(test_path / str(task_id + '.json'))

   

    with open(f, 'r') as read_file:

        task = json.load(read_file)

    

    n = len(task['train'])

    Input = [Defensive_Copy(task['train'][j]['input']) for j in range(n)]

    Output = [Defensive_Copy(task['train'][j]['output']) for j in range(n)]

    Input.append(Defensive_Copy(task['test'][pair_id]['input']))

    

    solution = Recolor([Input, Output])

   

    

    pred = ''

        

    if solution != -1:

        Solved.append(i)

        pred1 = flattener(solution)

        pred = pred+pred1+' '

        

    if pred == '':

        pred = flattener(example_grid)

        

    Proposed_Answers.append(pred)

    

sample_sub2['output'] = Proposed_Answers

sample_sub2.to_csv('submission2.csv', index = False)
sample_sub1 = sample_sub1.reset_index()

sample_sub1 = sample_sub1.sort_values(by="output_id")



sample_sub2 = sample_sub2.sort_values(by="output_id")

out1 = sample_sub1["output"].astype(str).values

out2 = sample_sub2["output"].astype(str).values



merge_output = []

for o1, o2 in zip(out1, out2):

    o = o1.strip().split(" ")[:1] + o2.strip().split(" ")[:2]

    o = " ".join(o[:3])

    merge_output.append(o)

sample_sub1["output"] = merge_output

sample_sub1["output"] = sample_sub1["output"].astype(str)

sample_sub1.to_csv("submission.csv", index=False)