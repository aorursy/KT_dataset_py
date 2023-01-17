import json

import matplotlib.pyplot as plt

import statistics



player_reward = []

num_ship = []

zero_hal = []

cargo = []

def calculate_stats(filename):

    global player_reward

    global num_ship

    global zero_hal

    global cargo

    with open(filename, "r") as read_file:

        dev = json.load(read_file)

        player_reward.clear()

        num_ship.clear()

        zero_hal.clear()

        cargo.clear()

        for player in range(0,4):

            player_reward.append([])

            num_ship.append([])

            zero_hal.append([])

            cargo.append([])

        for step in range(0,400):

            for player in range(0,4):

                player_reward[player].append(dev["steps"][step][player]["reward"])

                num_ship[player].append(len(dev["steps"][step][0]["observation"]["players"][player][2]))

                #print(f' Step {step} Player: {player} H: {dev["steps"][step][player]["reward"]}'

                #      f' ship: {len(dev["steps"][step][0]["observation"]["players"][player][2])}')

                sl = dev["steps"][step][0]["observation"]["players"][player][2]

                val_list = [k for k, v in sl.items() if v[1] == 0]

                # print(f' H0 list: {val_list}')

                zero_hal[player].append(len(val_list))

                cg = 0

                for v in sl.values():

                    cg += v[1]

                cargo[player].append(cg)



def plot(filename):

    global player_reward

    global num_ship

    global zero_hal

    calculate_stats(filename)



    fig, ax1 = plt.subplots(figsize=(12,4.5))

    ax1.set_ylabel('Halite')

    ax1.set_xlabel('Step')

    ax1.plot(player_reward[0],'y')

    ax1.plot(player_reward[1],'r')

    ax1.plot(player_reward[2],'g')

    ax1.plot(player_reward[3],'m')

    ax1.axis([0, 400, 0, 10000])

    ax1.set_title('Reward')



    fig, ax2 = plt.subplots(figsize=(12,4.5))

    ax2.set_ylabel('Ship')

    ax2.set_xlabel('Step')

    ax2.plot(num_ship[0],'y')

    ax2.plot(num_ship[1],'r')

    ax2.plot(num_ship[2],'g')

    ax2.plot(num_ship[3],'m')

    ax2.axis([0, 400, 0, 40])

    ax2.set_title('No. of Ships over time')



    fig, ax3 = plt.subplots(figsize=(12,4.5))

    ax3.set_ylabel('Ship')

    ax3.set_xlabel('Step')

    ax3.plot(zero_hal[0],'y')

    ax3.plot(zero_hal[1],'r')

    ax3.plot(zero_hal[2],'g')

    ax3.plot(zero_hal[3],'m')

    ax3.axis([0, 400, 0, 30])

    ax3.set_title('No. of Ships w/ 0 halite over time')

    plt.show()

    

    fig, ax4 = plt.subplots(figsize=(12,4.5))

    ax4.set_ylabel('Cargo')

    ax4.set_xlabel('Step')

    ax4.plot(cargo[0],'y')

    ax4.plot(cargo[1],'r')

    ax4.plot(cargo[2],'g')

    ax4.plot(cargo[3],'m')

    ax4.axis([0, 400, 0, 3000])

    ax4.set_title('Cargo of Ships over time')

    plt.show()
plot('../input/leu-mzot/2473070_leu_yellow_mzot_red.json')
plot('../input/leu-mzot/2478393_leu_red_mzot_green.json')
plot('../input/convexdata/2947984_convex_g1_rage_y2_el_p3_fei_r4.json')