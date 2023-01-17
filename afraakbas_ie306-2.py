!pip install simpy
import simpy
import random
import numpy as np


INTERARRIVAL_RATE = 1/6     # assuming random.expovariate takes lambda as 1/mean
AUTOMATED_RECORD_RATE = 1/5           #
SERVICE_RANGE = [1, 7]


service_times = [0, 0] #Duration of the conversation between the customer and the operator (Service time)
queue_w_times = [] #Time spent by a customer while it waits for the operator (Queue waiting time Wq)
queue_data = [[], []]    # list of [t,customer_num_in_queue] for each entrance and leaving to the queue
customer_in_queue = 0
customer_number_in_system = 0 # customer number waiting in the system
remaining_customers = 0  # remaining customers to end the simulation until the end
unsatified_customers = 0
total_waiting_time = 0
total_time_in_system = 0
ans_total_busy = 0
ans_busy_time = 0


class Customer(object):
    def __init__(self, name, env, opr):
        self.env = env
        self.name = name
        self.arrival_t = self.env.now
        self.action = env.process(self.call())

    def call(self):
        global customer_number_in_system
        global remaining_customers
        global unsatified_customers
        global total_time_in_system
        global total_waiting_time
        global ans_busy_time
        global ans_total_busy
        
        
        if customer_number_in_system == 0:
            ans_busy_time = self.env.now
            
        customer_number_in_system += 1

        env.timeout(random.expovariate(AUTOMATED_RECORD_RATE))

        customer_number_in_system -= 1
        if customer_number_in_system == 0:
            ans_total_busy += self.env.now - ans_busy_time
        
        r = random.randint(1,10)
        if r <= 7:
            self.opr_num = 1
        else:
            self.opr_num = 0
        
        r = random.random()
        if r <= 0.1:
            # There will be some things for statistical purposes
            customer_number_in_system -= 1
            remaining_customers -= 1
            unsatified_customers += 1
            return
        
        with operators[self.opr_num].request(priority=1, preempt=False) as req:
            
            queue_data[self.opr_num].append([self.env.now, len(operators[0].queue)+len(operators[1].queue)])
    
            
            yield req 
            waiting_time = self.env.now - self.arrival_t
            if waiting_time >= 10:
                queue_w_times.append(10)
                total_waiting_time += 10   #assume unsatisfied customers included
            else:
                queue_w_times.append(waiting_time)
                total_waiting_time += waiting_time
            
            queue_data[self.opr_num].append([self.env.now, len(operators[0].queue)+len(operators[1].queue)])
            
            try:
                
                if (waiting_time) < 10:   # the customer is in next phase
                    yield self.env.process(self.ask_question())
                    total_time_in_system += self.env.now - self.arrival_t
                else:
                    unsatified_customers += 1
                    total_time_in_system += 10   #assume unsatisfied customers included
            except:
                print('something wrong')
            
            
            remaining_customers -= 1

    def ask_question(self):
        if self.opr_num == 0:
            duration = np.random.lognormal(2.37,0.22)
        else:
            duration = random.uniform(*SERVICE_RANGE)
            
        yield self.env.timeout(duration)
        service_times[self.opr_num] += duration
        
        

def customer_generator(env, operators, size):
    for i in range(size):
        yield env.timeout(random.expovariate(INTERARRIVAL_RATE))
        if customer_number_in_system < 100:
            customer = Customer('Cust %s' %(i+1), env, operators)
        else:
            # There might be some things for statistical purposes
            remaining_customers -= 1

def initiate_a_break(env, operator, opr_num, breaks):
    with operator.request(priority=2, preempt=False) as req:
        yield req
        yield env.timeout(3)

def decide_break_times(env, operators):
    for operator, num in zip(operators, range(2)):
        # tek bir shift için
        break_number = np.random.poisson(5)
        break_durations = [np.random.randint(1, 480) for x in range(break_number)]
        
        for i in range(break_number):
            yield env.timeout(break_durations[i])
            env.process(initiate_a_break(env, operator, num, i))
            
def calculate_WQ(queue_data, total):
    area = [0, 0]
    for i in range(2):
        dt = 0
        previous_t = 0
        for data_point in queue_data[i]:
            dt = data_point[0] - previous_t
            previous_t = data_point[0]
            area[i] += dt * data_point[1]
            
    return area[0]/total, area[1]/total

  
def take_average(data):
    average = sum(data) / len(data)
    return average

def take_avg_list(data):
    summation = [0 for x in range(8)]
    for replication in data:
        for i in range(8):
            summation[i] += replication[i]
    
    length = len(data)
    avg_data = [x/length for x in summation]

    return avg_data
        
result = [[],[]]
avg_result = [0, 0]
size = [1000, 5000]
seeds = random.sample(range(1000),10)
for seed in seeds:
    RANDOM_SEED = seed
    random.seed(RANDOM_SEED)
    for s, j in zip(size, range(2)):
        remaining_customers = s
        env = simpy.Environment()
        operators = [simpy.PriorityResource(env, capacity = 1), simpy.PriorityResource(env, capacity = 1)]
        env.process(decide_break_times(env, operators))
        env.process(customer_generator(env, operators, s))
        env.run()
        total_t = env.now
        utilization_ans = ans_total_busy / total_t
        utilization_opr_1 = service_times[0] / total_t
        utilization_opr_2 = service_times[1] / total_t
        average_total_t = take_average(queue_w_times)
        waiting_system_ratio = total_waiting_time / total_time_in_system
        avg_cust_num_opr_1, avg_cust_num_opr_2 = calculate_WQ(queue_data, total_t) 
        avg_unsatisfied = unsatified_customers

        result[j].append([utilization_ans, utilization_opr_1, utilization_opr_2, average_total_t, 
                          waiting_system_ratio, avg_cust_num_opr_1, avg_cust_num_opr_2, avg_unsatisfied])

        queue_w_times = []
        service_times = [0, 0]
        customer_in_queue = 0
        remaining_customers = 0
        unsatified_customers = 0
        total_waiting_time = 0
        total_time_in_system = 0
        utilization_ans = 0
        utilization_opr_1 = 0
        utilization_opr_2 = 0
        
        queue_data = [[], []]
        customer_number_in_system = 0

avg_result[0] = take_avg_list(result[0])
avg_result[1] = take_avg_list(result[1])
print(avg_result[0], avg_result[1])
