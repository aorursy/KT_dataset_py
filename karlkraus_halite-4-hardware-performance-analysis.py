import matplotlib.pyplot as plt

plt.figure(tight_layout=True, figsize=(12,4))
for i, time in enumerate([13.70, 18.60, 2.7921 * 10, 42, 7 * 10]):
    plt.bar(i, 420e3 * 350 / time, .5)
plt.gca().set_xticks(range(5))
plt.gca().set_xticklabels(["workstation", "laptop", "codeforces\n(extrapolated)", "notebook", "online\n(extrapolated, optimistic)"])
plt.title("append(random()) on different hardware")
plt.ylabel("iterations/s")
plt.grid()
plt.show()
import kaggle_environments

# patch renderer: https://github.com/Kaggle/kaggle-environments/pull/55
old_html = kaggle_environments.envs.halite.halite.html_renderer()
patched_html = old_html.replace(".action;", ".action || {};")
def evaluate(agent):
    environment = kaggle_environments.make("halite")
    environment.html_renderer = lambda: patched_html
    environment.run([agent] * 4)
    environment.render(mode="ipython")
    return environment

def filter_actions(actions):
	return { id: action for id, action in actions.items() if action is not None }

def run_base(act):
    def run(observation):
        step = observation["step"]
        player = observation["player"]
        players = observation["players"]
        my_halite, my_shipyards, my_ships = players[player]
        if step == 0:
            return { id: "CONVERT" for id in my_ships }
        if step < 10:
            return {
                **{ id: "SPAWN" for id in my_shipyards },
                **{ id: "EAST" for id in my_ships },
            }
        if step < 350 + player:
            return filter_actions(act(step, player, my_ships))
        raise Exception(42)
    return run

def idle(step, player, my_ships):
    return {}

idle_steps = evaluate(run_base(idle)).steps

def bit(id):
	return int(id.split("-")[0]) - 2

def display(position, value):
	y = position // 21
	target = y & 1
	if target == value:
		return None
	elif target == 0:
		return "NORTH"
	elif target == 1:
		return "SOUTH"

def run_binary(measure):
    def run(step, player, my_ships):
        quantized = int(measure(step, player))
        return { id: display(my_ships[id][0], (quantized >> bit(id)) & 1) for id in my_ships }
    return run_base(run)

import math
def dance(step, player):
    return step * (player + 1) + math.sin(step * math.pi / 10) * 10

dance_steps = evaluate(run_binary(dance)).steps
def value(position):
	y = position // 21
	target = y & 1
	return target

players = 4
start_step = 10
player_labels = lambda: plt.legend(["1", "2", "3", "4"])

def decode(steps):
    counts = [[None] * players] * start_step
    for step in steps[start_step + 1:]:
        count = []
        for player in step[0]["observation"]["players"]:
            ships = player[2]
            if len(ships) == 0:
                z = None
            else:
                z = 0
                for id in ships:
                    z += value(ships[id][0]) << bit(id)
            count.append(z)
        counts.append(count)
    return counts

counts = decode(dance_steps)
figure = plt.figure(tight_layout=True, figsize=(12, 4))
for player in range(4): # encoded counts
    steps = range(50, 301, 50)
    plt.scatter(steps, [dance(step, player) for step in steps])
plt.plot(counts) # decoded counts
player_labels()
plt.xlabel("step")
plt.ylabel("value")
plt.grid()
plt.show()

print("encoded", dance(255, 0), "decoded", counts[255][0])
def get_total(counts):
    unrolled_counts = list(counts[:start_step + 1])
    for last_count, count in zip(counts[start_step:-1], counts[start_step + 1:]):
        unrolled = []
        for total, y, z in zip(unrolled_counts[-1], last_count, count):
            if z is None or total is None:
                unrolled.append(None)
            else:
                delta = (z - y + 256) % 512 - 256
                unrolled.append(total + delta)
        unrolled_counts.append(unrolled)
    return unrolled_counts

total = get_total(counts)
figure = plt.figure(tight_layout=True, figsize=(12, 4))
for player in range(4): # encoded counts
    steps = range(50, 301, 50)
    plt.scatter(steps, [dance(step, player) for step in steps])
plt.plot(total) # decoded counts
player_labels()
plt.xlabel("step")
plt.ylabel("value")
plt.grid()
plt.show()

for i in [10, 11, 12, 255]:
    print(i, "encoded", dance(i, 3), "decoded", total[i][3])
for i in range(20):
    print(i, counts[i], total[i])
def get_deltas(counts):
    deltas = [[None] * players] * (start_step + 1)
    for last_count, count in zip(counts[start_step:-1], counts[start_step + 1:]):
        unrolled = []
        for y, z in zip(last_count, count):
            if z is None:
                unrolled.append(None)
            else:
                delta = (z - y + 256) % 512 - 256
                unrolled.append(delta)
        deltas.append(unrolled)
    return deltas

deltas = get_deltas(counts)
figure = plt.figure(tight_layout=True, figsize=(12, 4))
for player in range(4): # encoded counts
    steps = range(50, 301, 50)
    plt.scatter(steps, [dance(step, player) - dance(step - 1, player) for step in steps])
plt.plot(deltas) # decoded counts
player_labels()
plt.xlabel("step")
plt.ylabel("change")
plt.grid()
plt.show()

print(len(counts), len(total), len(deltas))
for i in [11, 12, 256]:
    print("encoded", int(dance(i, 3)) - int(dance(i-1, 3)), "decoded", deltas[i][3])
for i in range(20):
    print(i, counts[i], total[i], deltas[i])
import time

def test_timeout(step, player):
    now = time.perf_counter()
    if (step % 10) == 0:
        print(player, step, step / 50)
        time.sleep(step / 50)
    else:
        time.sleep(.15)
    return now * 10
import random

def test_iterations(iterations_per_step, players):
    def test(step, player):
        now = time.perf_counter()
        if player not in players:
            print(player, step, "%.5f" % now)
            return now * 10
        else:
            z = []
            for i in range(iterations_per_step * step):
                z.append(random.random())
            then = time.perf_counter()
            interval = then - now
            print(player, step, "%.5f" % now, "%.5f" % interval)
            return interval * 100
    return test

test_performance = test_iterations(420000, [1, 2])
test_performance_once = test_iterations(420000, [1])
test_performance_light = test_iterations(42000, [1, 2])
import time
import random

def final_loop():
    a = time.process_time()
    at = time.perf_counter()
    z = []
    for i in range(420000 * 350):
        z.append(random.random())
    del z
    b = time.process_time()
    bt = time.perf_counter()
    return b - a, bt - at

#final_loop()
import json

def fix_scale(data, scales):
    return [[None if x is None else x * scale for x, scale in zip(values, scales)] for values in data]

def load_result(file, scales):
    with open(file, "r") as f:
        data = json.load(f)
    steps = data["steps"]
    counts = decode(steps)
    total = fix_scale(get_total(counts), scales)
    deltas = fix_scale(get_deltas(counts), scales)
    return total, deltas

result_labels = ["workstation", "laptop", "notebook", "online"]
results_timeout = [load_result(file, [1/10, 1/10, 1/10, 1/10]) for file in [
    "/kaggle/input/halite-4-hardware-performance/local_test_timeout.json",
    "/kaggle/input/halite-4-hardware-performance/laptop_test_timeout.json",
    "/kaggle/input/halite-4-hardware-performance/notebook_test_timeout.json",
    "/kaggle/input/halite-4-hardware-performance/1577246.json",
]]
results_performance = [load_result(file, [1/10, 1/100, 1/100, 1/10]) for file in [
    "/kaggle/input/halite-4-hardware-performance/local_test_performance.json",
    "/kaggle/input/halite-4-hardware-performance/laptop_test_performance.json",
    "/kaggle/input/halite-4-hardware-performance/notebook_test_performance.json",
    "/kaggle/input/halite-4-hardware-performance/1577603.json",
]]
results_performance_once = [load_result(file, [1/10, 1/100, 1/10, 1/10]) for file in [
    "/kaggle/input/halite-4-hardware-performance/local_test_performance_once.json",
    "/kaggle/input/halite-4-hardware-performance/laptop_test_performance_once.json",
    "/kaggle/input/halite-4-hardware-performance/notebook_test_performance_once.json",
    "/kaggle/input/halite-4-hardware-performance/1578314.json",
]]
results_performance_light = [load_result(file, [1/10, 1/100, 1/100, 1/10]) for file in [
    "/kaggle/input/halite-4-hardware-performance/local_test_performance_light.json",
    "/kaggle/input/halite-4-hardware-performance/laptop_test_performance_light.json",
    "/kaggle/input/halite-4-hardware-performance/notebook_test_performance_light.json",
    "/kaggle/input/halite-4-hardware-performance/1581528.json",
]]
for title, (total, deltas) in zip(result_labels, results_timeout):
    plt.figure(tight_layout=True, figsize=(12, 4))
    plt.title(title)
    plt.grid()
    plt.plot(deltas)
    player_labels()
    plt.axhline(23.2, color="black")
plt.show()

def pick(data, players):
    return [[x if player in players else None for player, x in enumerate(values)] for values in data]

def inverse_pick(data, players):
    return [[x if player not in players else None for player, x in enumerate(values)] for values in data]

def show_performance(results, players):
    for title, (total, deltas) in zip(result_labels, results):
        plt.figure(tight_layout=True, figsize=(24, 4))
        plt.subplot(121)
        plt.title(title)
        plt.grid()
        plt.plot([None if values[0] is None or values[3] is None else values[3] - values[0] for values in total])
        plt.legend(["start(4)-start(1)"])
        plt.subplot(122)
        plt.title(title)
        plt.grid()
        plt.plot(pick(total, players))
        player_labels()
    plt.show()

show_performance(results_performance, [1, 2])
show_performance(results_performance_once, [1])
show_performance(results_performance_light, [1, 2])
def show_iterations_per_second(results, players, ops):
    results_speeds = []
    for total, delta in results:
        speeds = []
        for step, values in enumerate(total):
            for player in players:
                interval = values[player]
                if interval is not None:
                    speeds.append(ops * step / interval)
        results_speeds.append(speeds)
    plt.figure(tight_layout=True, figsize=(12, 4))
    plt.xlim(0, 1.5e7)
    for speeds in results_speeds:
        plt.hist(speeds, histtype="step", bins=10, weights=[1/len(speeds)]*len(speeds))
    plt.legend(result_labels)
    plt.xlabel("iterations/s")
    for speeds in results_speeds:
        plt.axvline(sum(speeds) / len(speeds), color="black")
        plt.axvline(min(speeds), color="red")
    plt.grid()
    plt.show()
    print(result_labels)
    print([sum(x)/len(x) for x in results_speeds])
    print([min(x) for x in results_speeds])
    return results_speeds

ips_performance = show_iterations_per_second(results_performance, [1, 2], 420e3)
ips_performance_light = show_iterations_per_second(results_performance_light, [1, 2], 42e3)
# Evaluation in Notebook
run_notebook = False
if run_notebook:
    result = evaluate(run_binary(test_timeout)).render(mode="json")
    with open("/kaggle/working/notebook_test_timeout.json", "w") as f:
        f.write(result)
if run_notebook:
    result = evaluate(run_binary(test_performance)).render(mode="json")
    with open("/kaggle/working/notebook_test_performance.json", "w") as f:
        f.write(result)
if run_notebook:
    result = evaluate(run_binary(test_performance_once)).render(mode="json")
    with open("/kaggle/working/notebook_test_performance_once.json", "w") as f:
        f.write(result)
if run_notebook:
    result = evaluate(run_binary(test_performance_light)).render(mode="json")
    with open("/kaggle/working/notebook_test_performance_light.json", "w") as f:
        f.write(result)