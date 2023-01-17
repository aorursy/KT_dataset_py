import random
def next_bus(last_bus):
    r = random.random() # [0.0, 1.0]
    if r < 0.5: # 50% chance
        return last_bus + 5
    else:
        return last_bus + 10
def next_passanger(last_passanger):
    return last_passanger + 60*24
the_bus = 0
the_passanger = 0

wait_total = 0
num_waits = 0

print('wait_time (the_bus-previous_bus) the_passanger the_bus previous_bus')
for i in range(50):
    the_passanger = next_passanger(the_passanger)
    while the_bus <= the_passanger:
        previous_bus = the_bus
        the_bus = next_bus(the_bus)
    wait_time = the_bus - the_passanger
    print('{:9} {:22} {:13} {:7} {:12}'.format(wait_time, (the_bus-previous_bus), the_passanger, the_bus, previous_bus))
    wait_total += wait_time
    num_waits += 1
    
wait_average = wait_total / num_waits
print('Average wait time is {}'.format(wait_average))