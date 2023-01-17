import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
class gridworld:

    

    def __init__(self):

        self.dim = [5, 5]

        self.pos_A = [0, 1]

        self.rew_A = 10

        self.trans_A = [4, 1]

        self.pos_B = [0, 3]

        self.rew_B = 5

        self.trans_B = [2, 3]

        # Define starting position

        self.start = [4, 0]

        self.s = self.start[:]

        self.reward = 0

            

        # Step count

        self.n = 0

        self.action_space = ["U", "L", "D", "R"]

        self.action_prob = [0.25, 0.25, 0.25, 0.25]

    

    # Show empty environment

    def show_grid(self):

        # print rows

        for i in range(self.dim[0]):

            print("-" * (self.dim[0] * 5 + 1))

            row = []

            for j in range(self.dim[1]):

                if i == self.pos_A[0] and j == self.pos_A[1]:

                    row.append("| A ")

                elif i == self.pos_B[0] and j == self.pos_B[1]:

                    row.append("| B ")

                elif i == self.trans_A[0] and j == self.trans_A[1]:

                    row.append("| A'")

                elif i == self.trans_B[0] and j == self.trans_B[1]:

                    row.append("| B'")

                elif i == self.start[0] and j == self.start[1]:

                    row.append("| S ")

                else:

                    row.append("|   ")

            row.append("|  ")

            print(' '.join(row))

        print("-" * (self.dim[0] * 5 + 1))

        

    # Show state

    def show_state(self):

        # print rows

        for i in range(self.dim[0]):

            print("-" * (self.dim[0] * 5 + 1))

            row = []

            for j in range(self.dim[1]):

                if i == self.s[0] and j == self.s[1]:

                    row.append("| X ")

                elif i == self.pos_A[0] and j == self.pos_A[1]:

                    row.append("| A ")

                elif i == self.pos_B[0] and j == self.pos_B[1]:

                    row.append("| B ")

                elif i == self.trans_A[0] and j == self.trans_A[1]:

                    row.append("| A'")

                elif i == self.trans_B[0] and j == self.trans_B[1]:

                    row.append("| B'")

                else:

                    row.append("|   ")

            row.append("|  ")

            print(' '.join(row))

        print("-" * (self.dim[0] * 5 + 1))

        

    # Give the agent an action

    def action(self, a):

        if a not in self.action_space:

            return "Error: Invalid action submission"

        # Check for special transition states

        if self.s == self.pos_A:

            self.s = self.trans_A[:]

            self.reward = self.rew_A

        elif self.s == self.pos_B:

            self.s = self.trans_B[:]

            self.reward = self.rew_B

        # Move up

        elif a == "U" and self.s[0] > 0:

            self.s[0] -= 1

            self.reward = 0

        # Move left

        elif a == "L" and self.s[1] > 0:

            self.s[1] -= 1

            self.reward = 0

        # Move down

        elif a == "D" and self.s[0] < self.dim[0] - 1:

            self.s[0] += 1

            self.reward = 0

        # Move right

        elif a == "R" and self.s[1] < self.dim[1] - 1:

            self.s[1] += 1

            self.reward = 0

        else:

            self.reward = -1

        self.n += 1

        return self.s, self.reward

            

    def reset(self):

        self.s = self.start

        self.reward = 0

        self.n = 0
grid = gridworld()

grid.show_grid()
q = np.zeros((grid.dim[0], grid.dim[1], len(grid.action_space)))

gamma = 0.9

delta = 1e-5

delta_t = 1



while delta_t > delta:

    q_old = q.copy()

    for i in range(grid.dim[0]):

        for j in range(grid.dim[1]):

            for a in grid.action_space:

                grid.s = [i, j]

                s, r = grid.action(a)

                a_index = grid.action_space.index(a)

                q[i, j, a_index] = r + gamma * np.max(q_old[s[0], s[1]])

    delta_t = np.sum(np.abs(q - q_old))

    

print(np.max(q, axis=2).round(1))
def opt_policy(q, grid):

    q_max = np.max(q, axis=2)

    x = np.linspace(0, grid.dim[0] - 1, grid.dim[0]) + 0.5

    y = np.linspace(grid.dim[1] - 1, 0, grid.dim[1]) + 0.5

    X, Y = np.meshgrid(x, y)

    zeros = np.zeros((grid.dim))

    

    fig = plt.figure(figsize=(12,8))

    ax = plt.axes()

    

    for i, action in enumerate(grid.action_space):

        q_star = np.zeros((5, 5))

        for j in range(grid.dim[0]):

            for k in reversed(range(grid.dim[1])):

                if q[j, k, i] == q_max[j, k]:

                    q_star[j, k] = 0.4

        # Plot results

        if action == "U":

            # Vectors point in positive Y-direction

            plt.quiver(X, Y, zeros, q_star, scale=1, units='xy')

        elif action == "L":

            # Vectors point in negative X-direction

            plt.quiver(X, Y, -q_star, zeros, scale=1, units='xy')

        elif action == "D":

            # Vectors point in negative Y-direction

            plt.quiver(X, Y, zeros, -q_star, scale=1, units='xy')

        elif action == "R":

            # Vectors point in positive X-direction

            plt.quiver(X, Y, q_star, zeros, scale=1, units='xy')

        

    plt.xlim([0, grid.dim[0]])

    plt.ylim([0, grid.dim[1]])

    ax.set_yticklabels([])

    ax.set_xticklabels([])

    plt.grid()

    plt.show()
opt_policy(q, grid)