import random as rn

import matplotlib.pyplot as plt
class environment():

    def __init__(self,food_amount,max_movement,single_step_limit,creatures):

        self.food_amount = food_amount

        self.food_locations = {loc:0 for loc in range(self.food_amount)}

        self.max_movement = max_movement

        self.single_step_limit = single_step_limit

        self.creatures = creatures

        self.creature_number = 0

        self.population = 0

        self.population_per_day = list()

        self.latest_generation = 0
class dove:

    population = 0

    latest_generation = 0

    

    def __init__(self):

        self.name = ''

        self.energy = 1.0

        self.generation = 0

        self.location = 0

        self.nature = 'dove'

        self.is_hungry = True

        

    def go_home(self):

        self.location = 0

        

    def get_population():

        return dove.population

    

    def get_latest_generation():

        return dove.latest_generation

    

    def add_dove():

        dove.population += 1

        

    def kill_dove():

        dove.population -= 1
def populate(new_creatures,env_obj):

    for i in range(new_creatures):

        d = dove()

        d.name = 'dove' + str(env_obj.creature_number)

        env_obj.creature_number += 1

        d.generation = env_obj.latest_generation + 1

        env_obj.creatures.append(d)

        env_obj.population += 1

    env_obj.latest_generation += 1
def find_food(env_obj):

    c_index = 0

    for c in env_obj.creatures:

        while c.energy > 0 and c.is_hungry:

            c.location = rn.randint(0,env_obj.max_movement)

            c.energy -= rn.randint(0,env_obj.single_step_limit)/100

            if c.location < env_obj.food_amount:

                if env_obj.food_locations[c.location] < 2:

                    env_obj.food_locations[c.location] += 1

                    c.is_hungry = False

                    break

        else:

            if c.energy <= 0 and c.is_hungry:

                del env_obj.creatures[c_index]

                env_obj.population -= 1

        c_index += 1
def eat_food(env_obj):

    for c in env_obj.creatures:

        if env_obj.food_locations[c.location] == 1:

            c.energy = 2.0

            c.is_hungry = True

            c.go_home()

        else:

            c.energy = 1.0

            c.is_hungry = True

            c.go_home()
def reproduce(env_obj):

    offsprings = 0

    for c in env_obj.creatures:

        if c.energy == 2:

            c.energy = 1.0

            offsprings += 1

    else:

        populate(offsprings,env_obj)
def replenish_food(env_obj):

    env_obj.food_locations = {loc:0 for loc in range(env_obj.food_amount)}
def population_summary(env_obj):

    print("------------------------------------------------------------------------------------\nTotal population:",env_obj.population)

    print("\nLatest generation:",env_obj.latest_generation,"\n------------------------------------------------------------------------------------")

    

    i = 1

    for c in env_obj.creatures:

        print("Name",c.name,"\nNature:",c.nature,"\nGeneration:",c.generation,"\nEnergy:",c.energy,"\nLocation:",c.location,"\n---------------")

        i += 1
def simulate(env_obj,days,initial_population=3):

    global population_per_day

    populate(initial_population,env_obj)

    env_obj.population_per_day.append(env_obj.population)

    d = days

    day_list = [d for d in range(1,days+1)]

    

    while d > 1:

        find_food(env_obj)

        eat_food(env_obj)

        reproduce(env_obj)

        env_obj.population_per_day.append(env_obj.population)

        replenish_food(env_obj)

        d -= 1

    else:

        plt.figure(figsize=(16,10),dpi=80)

        plt.plot(day_list,env_obj.population_per_day,linestyle='-',marker='o')

        plt.show()  
env = environment(10,50,20,[])

env1 = environment(25,50,20,[])
simulate(env1,100)
simulate(env,100)