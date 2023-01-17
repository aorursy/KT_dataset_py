%matplotlib inline
import numpy as np



import time

import math



import matplotlib.pyplot as plt

from matplotlib import animation, rc



from IPython.display import HTML



import skimage.io
def case(s, samples):

    numbers = np.random.randint(-1000, 1000, size = s)

    numbers_list = numbers.tolist()



    total_py = 0

    for _ in range(samples):

        start = time.time()

        s = sum(numbers_list)

        end = time.time()

        total_py += end - start

    

    total_np = 0

    for _ in range(samples):

        start = time.time()

        s = np.sum(numbers)

        end = time.time()

        total_np += end - start

        

    return total_py / samples, total_np / samples

    

exps = []

py_times = []

np_times = []



for e in range(1, 9):

    exps.append(e)

    result = case(s = 10 ** e, samples = 10)

    py_times.append(result[0])

    np_times.append(result[1])
plt.title("Raw runtimes")

plt.plot(exps, py_times)

plt.plot(exps, np_times)

plt.legend(["Python", "Numpy"])

plt.xlabel("$\lg\ \mathrm{size}$")

plt.ylabel("Time [s]")

plt.show()
# Prevent division by zero

for i, t in enumerate(py_times):

    if t == 0: 

        np_times[0] = 0

        py_times[0] = 1



plt.title("Slowdown")

plt.plot(exps, np.array(np_times) / np.array(py_times))

plt.xlabel("$\lg\ \mathrm{size}$")

plt.ylabel("Times slower by using python")

plt.show()
x = np.array([2, 3, 4])

y = np.array([5, -2, 3])

print("x.y =", str(x.dot(y)))



print("x.shape:", x.shape)

print("y.shape:", y.shape)

print("x:", x)

print("x transpose:", x.T)
x = np.array([[2, 3, 4]]) # Row vector

y = np.array([[5], [-2], [3]]) # Column vector



print("x.shape:", x.shape)

print("y.shape:", y.shape)

print("x.y:\n", x.dot(y)) # Dot product -> still looks like a matrix

print("y.x:\n", y.dot(x)) # Outer product -> matrix
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

    base = ax.quiver([0, 0], [0, 0], [], [], color = ["red", "blue"], units = "xy", scale = 1)

 

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

        base.set_UVC([x[0], y[0]], [x[1], y[1]])

        return base,



    return animation.FuncAnimation(fig, update, frames = frames, interval = delay, blit = True, repeat = True)

# Identity



# I modified the code above to not use Affine2D, so we don't need a 3x3 matrix.

# Instead we get a np.array from the matrix and we multiply it with a point to get the transformed point.

# I also decided to add a fancy animation :3



# Ok, so writing the animation code was fun...

# Needless to say it took many hours of browsing through docs and stackoverflow.

# (THIS TOOK ME 8 WHOLE HOURS I'M NOT EVEN JOKING)

# Though at the end it was worth it. It looks awesome.

# There is a lot lot more that can be done!

#  e.g. specifying the grid at the beginning, make the colors more beautiful etc.



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
def read_image(url):

    img = skimage.io.imread(url)

    return img
cat_image_url = "https://d17fnq9dkz9hgj.cloudfront.net/uploads/2012/11/140272627-grooming-needs-senior-cat-632x475.jpg"

cat_image = read_image(cat_image_url)
cat_image[0][0] # First pixel
plt.imshow(cat_image)

plt.axis("off")

plt.show()
cat_image_r, cat_image_g, cat_image_b = [cat_image[:, :, i] for i in range(3)]

f, (ax_r, ax_g, ax_b) = plt.subplots(1, 3, figsize = (10, 5))

ax_r.imshow(cat_image_r, cmap = "gray")

ax_r.set_title("Red channel")

ax_g.imshow(cat_image_g, cmap = "gray")

ax_g.set_title("Green channel")

ax_b.imshow(cat_image_b, cmap = "gray")

ax_b.set_title("Blue channel")

plt.setp([ax_r, ax_g, ax_b], xticks = [], yticks = []) # Remove axis ticks

plt.show()
cat_image_r_normalized, cat_image_g_normalized, cat_image_b_normalized = [

    channel / 255 for channel in [cat_image_r, cat_image_g, cat_image_b]

] 

cat_image_gray = cat_image_r_normalized + cat_image_g_normalized + cat_image_b_normalized  

plt.imshow(cat_image_gray, cmap = "gray")

plt.title("Average grayscale image")

plt.show()
cat_image_gray_corrected = (0.299 * cat_image_r_normalized + 

                            0.587 * cat_image_g_normalized + 

                            0.114 * cat_image_b_normalized)

plt.gca().imshow(cat_image_gray, cmap = plt.cm.gray)

plt.title("Gamma-corrected grayscale image")

plt.show()
plt.hist(cat_image_gray.ravel(), bins = 256, color = "black")

plt.title("Uncorrected image histogram")

plt.show()

plt.hist(cat_image_gray_corrected.ravel(), bins = 256, color = "red")

plt.title("Corrected image histogram")

plt.show()
# Represent the image as a single-dimensional vector

hist_vector = cat_image_gray.ravel()



# Normalize the image to have values in the range [0; 1]

hist_vector = hist_vector / (hist_vector.max() - hist_vector.min())



plt.hist(hist_vector, bins = 256, color = "black", alpha = 0.5, label = "Uncorrected")

plt.hist(cat_image_gray_corrected.ravel(), bins = 256, color = "red", alpha = 0.5, label = "Corrected")

plt.xlim(0, 1)

plt.title("Image histograms comparison")

plt.legend()

plt.show()
# That's an image I took myself btw



ndk_url = "https://scontent.fsof5-1.fna.fbcdn.net/v/t1.15752-9/91288912_636373273869243_1748970013609426944_n.jpg?_nc_cat=108&_nc_sid=b96e70&_nc_ohc=gv_z9PDQGnEAX8o0YFI&_nc_ht=scontent.fsof5-1.fna&oh=7fde16a1a0edce51bb14246d4d0be19f&oe=5EA64BAB"

ndk_image = read_image(ndk_url)
plt.imshow(ndk_image)

plt.axis("off")

plt.show()
# Split channels

ndk_r, ndk_g, ndk_b = [ndk_image[:, :, i] for i in range(3)]

f, (ax_r, ax_g, ax_b) = plt.subplots(1, 3, figsize = (10, 5))

ax_r.imshow(ndk_r, cmap = "gray")

ax_r.set_title("Red channel")

ax_g.imshow(ndk_g, cmap = "gray")

ax_g.set_title("Green channel")

ax_b.imshow(ndk_g, cmap = "gray")

ax_b.set_title("Blue channel")

plt.setp([ax_r, ax_g, ax_b], xticks = [], yticks = []) # Remove axis ticks

plt.show()
ndk_r_norm, ndk_g_norm, ndk_b_norm = [

    channel / 255 for channel in [ndk_r, ndk_g, ndk_b]

] 

ndk_gray = ndk_r_norm + ndk_g_norm + ndk_b_norm  

plt.imshow(ndk_gray, cmap = "gray")

plt.title("Average grayscale image")

plt.show()
# Now we apply the gamma correction

ndk_corrected = (0.299 * ndk_r_norm + 

                 0.587 * ndk_g_norm + 

                 0.114 * ndk_b_norm)

plt.gca().imshow(ndk_corrected, cmap = plt.cm.gray)

plt.title("Gamma-corrected grayscale image")

plt.show()



# The whole image is a bit darker because the original has quite a low of blue in it. 
hist_vector = ndk_gray.ravel()

hist_vector = hist_vector / (hist_vector.max() - hist_vector.min())



plt.hist(hist_vector, bins = 256, color = "black", alpha = 0.5, label = "Uncorrected")

plt.hist(ndk_corrected.ravel(), bins = 256, color = "red", alpha = 0.5, label = "Corrected")

plt.xlim(0, 1)

plt.xlabel("Gamma")

plt.ylabel("Pixel count")

plt.title("Image histograms comparison")

plt.legend()

plt.show()
plt.hist(ndk_gray.ravel(), bins = 256, color = "black")

plt.title("Uncorrected image histogram")

plt.show()

plt.hist(ndk_corrected.ravel(), bins = 256, color = "red")

plt.title("Corrected image histogram")

plt.show()
# Write your code here
def visualize_transformed_vector(matrix, vec, title):

    """

    Shows the vector (starting at (0; 0)) before and after the transformation

    given by the specified matrix

    """

    # Write your code here

    pass
matrix = np.array([[2, -4, 0], [-1, -1, 0], [0, 0, 1]])

visualize_transformed_vector(matrix, [2, 3], "Transformation")
visualize_transformed_vector(matrix, [-4, 1], "Transformation")