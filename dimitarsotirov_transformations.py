%matplotlib inline
import numpy as np



import matplotlib.pyplot as plt

from matplotlib import animation, rc



from IPython.display import HTML
def gen_grid(ax):

    [min_x, max_x, min_y, max_y] = [2 * ax.get_xlim()[0], 2 * ax.get_xlim()[1] + 1, 2 * ax.get_ylim()[0], 2 * ax.get_ylim()[1] + 1]



    # Horizontal

    r = []

    for y in np.arange(min_y, max_y):

        r.append([min_x, y])

        r.append([max_x, y])

    h = np.column_stack(r)



    # Vertical

    r = []

    for x in np.arange(min_x, max_x):

        r.append([x, min_y])

        r.append([x, max_y])

    v = np.column_stack(r)

    return h, v



def visualize_transformation(A, title = None, frames = 10, delay = 50):

    """

    Creates an animated visualization of a transform specified by `A`

    """

    A = np.array(A)

    

    # Needed for Jupyter Notebook

    plt.rcParams["animation.html"] = "jshtml"

    

    fig, ax = plt.subplots()

    ax.set_aspect("equal")



    ax.set_title(title)

    ax.set_xlim(-5, 5)

    ax.set_ylim(-5, 5)

    ax.set_xticks(np.arange(ax.get_xlim()[0], ax.get_xlim()[1] + 1))

    ax.set_yticks(np.arange(ax.get_ylim()[0], ax.get_ylim()[1] + 1))

    ax.set_xticklabels([])

    ax.set_yticklabels([])

    plt.grid()

    

    # Draw the untransformed unit vectors

    ax.quiver([0, 0], [0, 0], [1, 0], [0, 1], color = ["red", "blue"], alpha = 0.2, units = "xy", scale = 1)



    h, v = gen_grid(ax)



    # Creates a list of intermediate h/v lines, stepping slowly from untransformed to the fully transformed ones.

    # We cache the lines instead of using the intermediate matrix to transform them later when drawing (takes a lot of time).

    # For the unit vectors we just use the matrix since it's just one point.

    hlines = []

    vlines = []

    intermediate_mats = []

    for i in range(frames + 1):

        t = np.eye(2) + i / frames * (A - np.eye(2))

        # We transform the end points using our transformation matrix and plot a line connecting them.

        # We can do this since we know that a linear transformation doesn't curve lines.

        hlines.append(t @ h)

        vlines.append(t @ v)

        intermediate_mats.append(t)

    

    # We need to save the state for the lines/vectors we will draw.

    # This is just how matplotlb animations work

    hplot = [None] * h.shape[1]

    vplot = [None] * v.shape[1]

    for i in range(len(hplot)):

        hplot[i] = ax.plot([], [], color = "red", linestyle = "--", linewidth = 2)[0]

    for i in range(len(vplot)):

        vplot[i] = ax.plot([], [], color = "blue", linestyle = "--", linewidth = 2)[0]

    basis = ax.quiver([0, 0], [0, 0], [], [], color = ["red", "blue"], units = "xy", scale = 1)

 

    # Close the final plot so we just see the animation

    # If you don't see the animation for some reason you need to comment this line

    # but call `update` below at least once and draw the final transformation 

    # (the final elements contain the fully transformed grid)

    plt.close()

    

    def update(i):

        h = hlines[i]

        v = vlines[i]

        t = intermediate_mats[i]



        hx = h[0, :]

        hy = h[1, :]

        for i in range(0, len(hplot), 2):

            hplot[i].set_data([hx[i], hx[i + 1]], [hy[i], hy[i + 1]])

            

        vx = v[0, :]

        vy = v[1, :]

        for i in range(0, len(vplot), 2):

            vplot[i].set_data([vx[i], vx[i + 1]], [vy[i], vy[i + 1]])

    

        # New (transformed) unit vectors

        x = np.dot(t, [1, 0])

        y = np.dot(t, [0, 1])

        basis.set_UVC([x[0], y[0]], [x[1], y[1]])

        return basis,



    return animation.FuncAnimation(fig, update, frames = frames, interval = delay, blit = True, repeat = True)

# Identity



# I modified the code to not use Affine2D, so we don't need a 3x3 matrix

matrix = [

    [1, 0],

    [0, 1]

]

visualize_transformation(matrix, r"$\mathrm{Identity\ transformation}$")
# Scaling

matrix = [

    [2, 0],

    [0, 1]

]

visualize_transformation(matrix, r"$\mathrm{Scaling}$", frames = 20, delay = 50)
# Shear

matrix = [

    [1, 2],

    [-1, 1]

]

visualize_transformation(matrix, r"$\mathrm{Shear}$", frames = 20, delay = 50)
# Rotation

matrix = [

    [np.cos(np.radians(30)), -np.sin(np.radians(30))],

    [np.sin(np.radians(30)), np.cos(np.radians(30))]

]

visualize_transformation(matrix, r"$\mathrm{30^{\circ}\ rotation}$", frames = 20, delay = 50)
# Projection (linearly dependent rows)

matrix = [

    [1, 2],

    [2, 4]

]

visualize_transformation(matrix, r"$\mathrm{Projection\ (linearly\ dependent\ rows)}$", frames = 100, delay = 50)
# Some flipping action

matrix = [

    [-2, 3],

    [1, 7]

]

visualize_transformation(matrix, r"$\mathrm{Some\ flipping\ action}$", frames = 50, delay = 50)
# Rotate, then scale

A = [

    [np.cos(np.radians(30)), -np.sin(np.radians(30))],

    [np.sin(np.radians(30)), np.cos(np.radians(30))]

]



B = [

    [2, 0],

    [0, 1]

]



visualize_transformation(np.dot(A, B), r"$\mathrm{Rotate\ 30\ degrees\ and\ scale}$", frames = 50, delay = 50)


