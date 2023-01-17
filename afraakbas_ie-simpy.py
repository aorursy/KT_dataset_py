!pip install simpy
import simpy
import random
import numpy as np


INTERARRIVAL_RATE = 1/30     # assuming random.expovariate takes lambda as 1/mean
AUTOMATED_RECORD_RATE = 1/5           #
SERVICE_RANGE = [1, 7]


service_times = [] #Duration of the conversation between the customer and the operator (Service time)
queue_w_times = [] #Time spent by a customer while it waits for the operator (Queue waiting time Wq)
queue_data = []    # list of [t,customer_num_in_queue] for each entrance and leaving to the queue
customer_in_queue = 0
customer_number_in_system = 0 # customer number waiting in the system
remaining_customers = 0  # remaining customers to end the simulation until the end
m = simpy.Monitor(name='LQ')

class Customer(object):
    def __init__(self, name, env, opr):
        self.env = env
        self.name = name
        self.arrival_t = self.env.now
        self.action = env.process(self.call())

    def call(self):
        global customer_number_in_system
        global remaining_customers

        customer_number += 1
        print('%s initiated a call at %g' % (self.name, self.env.now))

        env.timeout(random.expovariate(AUTOMATED_RECORD_RATE))

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
            print(self.name, 'dead')
            return

        with operators[self.opr_num].request(priority=1, preempt=False) as req:
            
            #if operators[self.opr_num].count > 0:
                #queue_data.append([self.now, len(operators[0].waitQ) + len(operators[1].waitQ))
                
                
            yield req #or self.env.timeout(10)
            #print('%s is assigned to an operator at %g' % (self.name, self.env.now))
            waiting_time = self.env.now - self.arrival_t
            queue_w_times.append(waiting_time)
            

            try:
                if (waiting_time) < 10:   # the customer is in next phase
                    yield self.env.process(self.ask_question())
                    print('%s is done at %g' % (self.name, self.env.now))
                else:
                    print(self.name, ' arrives ', self.arrival_t, ' and hangs up ', self.env.now)
            except:
                print('something wrong')

            customer_number_in_system -= 1
            remaining_customers -= 1

    def ask_question(self):
        if self.opr_num == 0:
            duration = np.random.lognormal(2.37,0.22)
        else:
            duration = random.uniform(*SERVICE_RANGE)
        #print(duration, ' is the dur of cust ', self.name, ' operator ', self.opr_num)
        yield self.env.timeout(duration)
        service_times.append(duration)
        

def customer_generator(env, operators, size):
    """Generate new cars that arrive at the gas station."""
    for i in range(size):
        yield env.timeout(random.expovariate(INTERARRIVAL_RATE))
        print(customer_number_in_system)
        if customer_number < 100:
            customer = Customer('Cust %s' %(i+1), env, operators)
        else:
            # There might be some things for statistical purposes
            print("The customer ",i," has left.")

def initiate_a_break(env, operator, opr_num, breaks):
    with operator.request(priority=2, preempt=False) as req:
        print(' decide to take a break', opr_num, ' bbb', breaks)
        yield req
        print('take a break')
        yield env.timeout(3)

def decide_break_times(env, operators):
    for operator, num in zip(operators, range(2)):
        # tek bir shift için
        break_number = np.random.poisson(5)
        break_durations = [np.random.randint(1, 480) for x in range(break_number)]
        print(break_number, ' it is the break num evet',)
        #if remaining_customers == 0:
         #   break
        for i in range(break_number):
            yield env.timeout(break_durations[i])
            env.process(initiate_a_break(env, operator, num, i))
            

for i in range(10):
    size = [1000,5000]
    seeds = random.sample(range(1000),10)
    for seed in seeds:
        RANDOM_SEED = seed
        random.seed(RANDOM_SEED)
        for s in size:
            remaining_customers = s
            env = simpy.Environment()
            operators = [simpy.PriorityResource(env, capacity = 1, monitored=True), simpy.PriorityResource(env, capacity = 1, monitored=True)]
            env.process(decide_break_times(env, operators))
            env.process(customer_generator(env, operators, s))
            env.run()
            print (queue_w_times)
            print (service_times)
            print(operators[0].waitMon)
            # operators[0].waitMon ile operators[1].waitMon birleştirme to get LQ
            customer_number_in_system = 0

