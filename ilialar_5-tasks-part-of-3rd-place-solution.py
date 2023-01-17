debug = False

fast_submit = False
import numpy as np



from matplotlib import pyplot as plt

import matplotlib as mpl





def show_sample(sample):

    """Shows the sample with tasks and answers"""

    print("Train:")

    for i in range(len(sample["train"])):

        fig = plt.figure()

        ax1 = fig.add_subplot(121)

        ax1.matshow(np.array(sample["train"][i]["input"]), cmap="Set3", norm=mpl.colors.Normalize(vmin=0, vmax=9))



        ax2 = fig.add_subplot(122)

        ax2.matshow(np.array(sample["train"][i]["output"]), cmap="Set3", norm=mpl.colors.Normalize(vmin=0, vmax=9))



        plt.show()



    print("Test:")

    for i in range(len(sample["test"])):

        fig = plt.figure()

        ax1 = fig.add_subplot(121)

        ax1.matshow(np.array(sample["test"][i]["input"]), cmap="Set3", norm=mpl.colors.Normalize(vmin=0, vmax=9))



        if "output" in sample["test"][i]:

            ax2 = fig.add_subplot(122)

            ax2.matshow(np.array(sample["test"][i]["output"]), cmap="Set3", norm=mpl.colors.Normalize(vmin=0, vmax=9))



        plt.show()





def matrix2answer(array):

    s = "|"

    for i in range(array.shape[0]):

        for j in range(array.shape[1]):

            s = s + str(int(array[i, j]))

        s = s + "|"

    return str(s)
import json

import time



import numpy as np



from scipy import ndimage

from scipy.stats import mode





def find_grid(image, frame=False, possible_colors=None):

    """Looks for the grid in image and returns color and size"""

    grid_color = -1

    size = [1, 1]



    if possible_colors is None:

        possible_colors = list(range(10))



    for color in possible_colors:

        for i in range(size[0] + 1, image.shape[0] // 2 + 1):

            if (image.shape[0] + 1) % i == 0:

                step = (image.shape[0] + 1) // i

                if (image[(step - 1) :: step] == color).all():

                    size[0] = i

                    grid_color = color

        for i in range(size[1] + 1, image.shape[1] // 2 + 1):

            if (image.shape[1] + 1) % i == 0:

                step = (image.shape[1] + 1) // i

                if (image[:, (step - 1) :: step] == color).all():

                    size[1] = i

                    grid_color = color



    if grid_color == -1 and not frame:

        color_candidate = image[0, 0]

        if (

            (image[0] == color_candidate).all()

            and (image[-1] == color_candidate).all()

            and (image[:, -1] == color_candidate).all()

            and (image[:, 0] == color_candidate).all()

        ):

            grid_color, size, _ = find_grid(

                image[1 : image.shape[0] - 1, 1 : image.shape[1] - 1], frame=True, possible_colors=[color_candidate]

            )

            return grid_color, size, frame

        else:

            return grid_color, size, frame



    return grid_color, size, frame





def find_color_boundaries(array, color):

    """Looks for the boundaries of any color and returns them"""

    if (array == color).any() == False:

        return None

    ind_0 = np.arange(array.shape[0])

    ind_1 = np.arange(array.shape[1])



    temp_0 = ind_0[(array == color).max(axis=1)]  # axis 0

    min_0, max_0 = temp_0.min(), temp_0.max()



    temp_1 = ind_1[(array == color).max(axis=0)]  # axis 1

    min_1, max_1 = temp_1.min(), temp_1.max()



    return min_0, max_0, min_1, max_1





def get_color_max(image, color):

    """Returns the part of the image inside the color boundaries"""

    boundaries = find_color_boundaries(image, color)

    if boundaries:

        return (0, image[boundaries[0] : boundaries[1] + 1, boundaries[2] : boundaries[3] + 1])

    else:

        return 1, None





def get_pixel(image, i, j):

    """Returns the pixel by coordinates"""

    if i >= image.shape[0] or j >= image.shape[1]:

        return 1, None

    return 0, image[i : i + 1, j : j + 1]





def get_pixel_fixed(image, i):

    return 0, np.array([[i]])





def get_grid(image, grid_size, cell, frame=False):

    """ returns the particular cell form the image with grid"""

    if frame:

        return get_grid(image[1 : image.shape[0] - 1, 1 : image.shape[1] - 1], grid_size, cell, frame=False)

    if cell[0] >= grid_size[0] or cell[1] >= grid_size[1]:

        return 1, None

    steps = ((image.shape[0] + 1) // grid_size[0], (image.shape[1] + 1) // grid_size[1])

    block = image[steps[0] * cell[0] : steps[0] * (cell[0] + 1) - 1, steps[1] * cell[1] : steps[1] * (cell[1] + 1) - 1]

    return 0, block





def get_half(image, side):

    """ returns the half of the image"""

    if side not in ["l", "r", "t", "b", "long1", "long2"]:

        return 1, None

    if side == "l":

        return 0, image[:, : (image.shape[1]) // 2]

    elif side == "r":

        return 0, image[:, -((image.shape[1]) // 2) :]

    elif side == "b":

        return 0, image[-((image.shape[0]) // 2) :, :]

    elif side == "t":

        return 0, image[: (image.shape[0]) // 2, :]

    elif side == "long1":

        if image.shape[0] >= image.shape[1]:

            return get_half(image, "t")

        else:

            return get_half(image, "l")

    elif side == "long2":

        if image.shape[0] >= image.shape[1]:

            return get_half(image, "b")

        else:

            return get_half(image, "r")





def get_corner(image, side):

    """ returns the half of the image"""

    if side not in ["tl", "tr", "bl", "br"]:

        return 1, None

    size = (image.shape[0]) // 2, (image.shape[1]) // 2

    if side == "tl":

        return 0, image[size[0] :, -size[1] :]

    if side == "tr":

        return 0, image[size[0] :, : size[1]]

    if side == "bl":

        return 0, image[: -size[0], : size[1]]

    if side == "br":

        return 0, image[: -size[0], -size[1] :]





def get_k_part(image, num, k):

    if image.shape[0] > image.shape[1]:

        max_axis = 0

        max_shape = image.shape[0]

    else:

        max_axis = 1

        max_shape = image.shape[1]



    if max_shape % num != 0:

        return 1, None

    size = max_shape // num



    if max_axis == 0:

        return 0, image[k * size : (k + 1) * size]

    else:

        return 0, image[:, k * size : (k + 1) * size]





def get_rotation(image, k):

    return 0, np.rot90(image, k)





def get_transpose(image):

    return 0, np.transpose(image)





def get_roll(image, shift, axis):

    return 0, np.roll(image, shift=shift, axis=axis)





def get_cut_edge(image, l, r, t, b):

    """deletes pixels from some sided of an image"""

    return 0, image[t : image.shape[0] - b, l : image.shape[1] - r]





def get_resize(image, scale):

    """ resizes image according to scale"""

    if isinstance(scale, int):

        if image.shape[0] % scale != 0 or image.shape[1] % scale != 0:

            return 1, None

        if image.shape[0] < scale or image.shape[1] < scale:

            return 2, None



        arrays = []

        size = image.shape[0] // scale, image.shape[1] // scale

        for i in range(scale):

            for j in range(scale):

                arrays.append(image[i::scale, j::scale])



        result = mode(np.stack(arrays), axis=0).mode[0]

    else:

        size = int(image.shape[0] / scale), int(image.shape[1] / scale)

        result = []

        for i in range(size[0]):

            result.append([])

            for j in range(size[1]):

                result[-1].append(image[int(i * scale), int(j * scale)])



        result = np.uint8(result)



    return 0, result





def get_resize_to(image, size_x, size_y):

    """ resizes image according to scale"""

    scale_x = image.shape[0] // size_x

    scale_y = image.shape[1] // size_y

    if scale_x == 0 or scale_y == 0:

        return 3, None

    if image.shape[0] % scale_x != 0 or image.shape[1] % scale_y != 0:

        return 1, None

    if image.shape[0] < scale_x or image.shape[1] < scale_y:

        return 2, None



    arrays = []

    for i in range(scale_x):

        for j in range(scale_y):

            arrays.append(image[i::scale_x, j::scale_y])



    result = mode(np.stack(arrays), axis=0).mode[0]

    if result.max() > 10:

        print(1)



    return 0, result





def get_reflect(image, side):

    """ returns images generated by reflections of the input"""

    if side not in ["r", "l", "t", "b", "rt", "rb", "lt", "lb"]:

        return 1, None

    try:

        if side == "r":

            result = np.zeros((image.shape[0], image.shape[1] * 2 - 1))

            result[:, : image.shape[1]] = image

            result[:, -image.shape[1] :] = image[:, ::-1]

        elif side == "l":

            result = np.zeros((image.shape[0], image.shape[1] * 2 - 1))

            result[:, : image.shape[1]] = image[:, ::-1]

            result[:, -image.shape[1] :] = image

        elif side == "b":

            result = np.zeros((image.shape[0] * 2 - 1, image.shape[1]))

            result[: image.shape[0], :] = image

            result[-image.shape[0] :, :] = image[::-1]

        elif side == "t":

            result = np.zeros((image.shape[0] * 2 - 1, image.shape[1]))

            result[: image.shape[0], :] = image[::-1]

            result[-image.shape[0] :, :] = image



        elif side == "rb":

            result = np.zeros((image.shape[0] * 2 - 1, image.shape[1] * 2 - 1))

            result[: image.shape[0], : image.shape[1]] = image

            result[: image.shape[0], -image.shape[1] :] = image[:, ::-1]

            result[-image.shape[0] :, : image.shape[1]] = image[::-1, :]

            result[-image.shape[0] :, -image.shape[1] :] = image[::-1, ::-1]



        elif side == "rt":

            result = np.zeros((image.shape[0] * 2 - 1, image.shape[1] * 2 - 1))

            result[: image.shape[0], : image.shape[1]] = image[::-1, :]

            result[: image.shape[0], -image.shape[1] :] = image[::-1, ::-1]

            result[-image.shape[0] :, : image.shape[1]] = image

            result[-image.shape[0] :, -image.shape[1] :] = image[:, ::-1]



        elif side == "lt":

            result = np.zeros((image.shape[0] * 2 - 1, image.shape[1] * 2 - 1))

            result[: image.shape[0], : image.shape[1]] = image[::-1, ::-1]

            result[: image.shape[0], -image.shape[1] :] = image[::-1, :]

            result[-image.shape[0] :, : image.shape[1]] = image[:, ::-1]

            result[-image.shape[0] :, -image.shape[1] :] = image



        elif side == "lb":

            result = np.zeros((image.shape[0] * 2 - 1, image.shape[1] * 2 - 1))

            result[: image.shape[0], : image.shape[1]] = image[:, ::-1]

            result[: image.shape[0], -image.shape[1] :] = image

            result[-image.shape[0] :, : image.shape[1]] = image[::-1, ::-1]

            result[-image.shape[0] :, -image.shape[1] :] = image[::-1, :]

    except:

        return 2, None



    return 0, result





def get_color_swap(image, color_1, color_2):

    """swapping two colors"""

    if not (image == color_1).any() and not (image == color_2).any():

        return 1, None

    result = image.copy()

    result[image == color_1] = color_2

    result[image == color_2] = color_1

    return 0, result





def get_cut(image, x1, y1, x2, y2):

    if x1 >= x2 or y1 >= y2:

        return 1, None

    else:

        return 0, image[x1:x2, y1:y2]





def get_min_block(image, full=True):

    if full:

        structure = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

    else:

        structure = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]

    masks, n_masks = ndimage.label(image, structure=structure)

    sizes = [(masks == i).sum() for i in range(1, n_masks + 1)]



    if n_masks == 0:

        return 2, None



    min_n = np.argmin(sizes) + 1



    boundaries = find_color_boundaries(masks, min_n)

    if boundaries:

        return (0, image[boundaries[0] : boundaries[1] + 1, boundaries[2] : boundaries[3] + 1])

    else:

        return 1, None





def get_min_block_mask(image, full=True):

    if full:

        structure = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

    else:

        structure = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]

    masks, n_masks = ndimage.label(image, structure=structure)

    sizes = [(masks == i).sum() for i in range(1, n_masks + 1)]



    if n_masks == 0:

        return 2, None



    min_n = np.argmin(sizes) + 1

    return 0, masks == min_n





def get_max_block_mask(image, full=True):

    if full:

        structure = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

    else:

        structure = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]

    masks, n_masks = ndimage.label(image, structure=structure)

    sizes = [(masks == i).sum() for i in range(1, n_masks + 1)]



    if n_masks == 0:

        return 2, None



    min_n = np.argmax(sizes) + 1

    return 0, masks == min_n





def get_max_block(image, full=True):

    if full:

        structure = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

    else:

        structure = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]

    masks, n_masks = ndimage.label(image, structure=structure)

    sizes = [(masks == i).sum() for i in range(1, n_masks + 1)]



    if n_masks == 0:

        return 2, None



    max_n = np.argmax(sizes) + 1



    boundaries = find_color_boundaries(masks, max_n)

    if boundaries:

        return (0, image[boundaries[0] : boundaries[1] + 1, boundaries[2] : boundaries[3] + 1])

    else:

        return 1, None





def get_block_with_side_colors(image, block_type="min", structure=0):

    if structure == 0:

        structure = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

    else:

        structure = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]

    masks, n_masks = ndimage.label(image, structure=structure)



    if n_masks == 0:

        return 2, None



    unique_nums = []

    for i in range(1, n_masks + 1):

        unique = np.unique(image[masks == i])

        unique_nums.append(len(unique))



    if block_type == "min":

        n = np.argmin(unique_nums) + 1

    else:

        n = np.argmax(unique_nums) + 1



    boundaries = find_color_boundaries(masks, n)

    if boundaries:

        return (0, image[boundaries[0] : boundaries[1] + 1, boundaries[2] : boundaries[3] + 1])

    else:

        return 1, None





def get_block_with_side_colors_count(image, block_type="min", structure=0):

    if structure == 0:

        structure = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

    else:

        structure = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]

    masks, n_masks = ndimage.label(image, structure=structure)

    if n_masks == 0:

        return 2, None



    unique_nums = []

    for i in range(1, n_masks + 1):

        unique, counts = np.unique(image[masks == i], return_counts=True)

        unique_nums.append(min(counts))



    if block_type == "min":

        n = np.argmin(unique_nums) + 1

    else:

        n = np.argmax(unique_nums) + 1



    boundaries = find_color_boundaries(masks, n)

    if boundaries:

        return (0, image[boundaries[0] : boundaries[1] + 1, boundaries[2] : boundaries[3] + 1])

    else:

        return 1, None





def get_color(color_dict, colors):

    """ retrive the absolute number corresponding a color set by color_dict"""

    for i, color in enumerate(colors):

        for data in color:

            equal = True

            for k, v in data.items():

                if k not in color_dict or v != color_dict[k]:

                    equal = False

                    break

            if equal:

                return i

    return -1





def get_mask_from_block(image, color):

    if color in np.unique(image, return_counts=False):

        return 0, image == color

    else:

        return 1, None





def get_background(image, color):

    return 0, np.uint8(np.ones_like(image) * color)





def get_mask_from_max_color_coverage(image, color):

    if color in np.unique(image, return_counts=False):

        boundaries = find_color_boundaries(image, color)

        result = (image.copy() * 0).astype(bool)

        result[boundaries[0] : boundaries[1] + 1, boundaries[2] : boundaries[3] + 1] = True

        return 0, image == color

    else:

        return 1, None





def add_unique_colors(image, result, colors=None):

    """adds information about colors unique for some parts of the image"""

    if colors is None:

        colors = np.unique(image)



    unique_side = [False for i in range(10)]

    unique_corner = [False for i in range(10)]



    half_size = (((image.shape[0] + 1) // 2), ((image.shape[1] + 1) // 2))

    for (image_part, side, unique_list) in [

        (image[: half_size[0]], "bottom", unique_side),

        (image[-half_size[0] :], "top", unique_side),

        (image[:, : half_size[1]], "right", unique_side),

        (image[:, -half_size[1] :], "left", unique_side),

        (image[-half_size[0] :, -half_size[1] :], "tl", unique_corner),

        (image[-half_size[0] :, : half_size[1]], "tr", unique_corner),

        (image[: half_size[0], : half_size[1]], "br", unique_corner),

        (image[: half_size[0], -half_size[1] :], "left", unique_corner),

    ]:

        unique = np.uint8(np.unique(image_part))

        if len(unique) == len(colors) - 1:

            color = [x for x in colors if x not in unique][0]

            unique_list[color] = True

            result["colors"][color].append({"type": "unique", "side": side})



    for i in range(10):

        if unique_corner[i]:

            result["colors"][i].append({"type": "unique", "side": "corner"})

        if unique_side[i]:

            result["colors"][i].append({"type": "unique", "side": "side"})

        if unique_side[i] or unique_corner[i]:

            result["colors"][i].append({"type": "unique", "side": "any"})



    return





def add_center_color(image, result, colors=None):

    i = image.shape[0] // 4

    j = image.shape[1] // 4

    center = image[i : image.shape[0] - i, j : image.shape[1] - j]

    values, counts = np.unique(center, return_counts=True)

    if len(counts) > 0:

        ind = np.argmax(counts)

        color = values[ind]

        result["colors"][color].append({"type": "center"})





def get_color_scheme(image, target_image=None, params=None):

    """processes original image and returns dict color scheme"""

    result = {

        "grid_color": -1,

        "colors": [[], [], [], [], [], [], [], [], [], []],

        "colors_sorted": [],

        "grid_size": [1, 1],

    }



    if params is None:

        params = ["coverage", "unique", "corners", "top", "grid"]



    # preparing colors info



    unique, counts = np.unique(image, return_counts=True)

    colors = [unique[i] for i in np.argsort(counts)]



    result["colors_sorted"] = colors

    result["colors_num"] = len(colors)



    for color in range(10):

        # use abs color value - same for any image

        result["colors"][color].append({"type": "abs", "k": color})



    if len(colors) == 2 and 0 in colors:

        result["colors"][[x for x in colors if x != 0][0]].append({"type": "non_zero"})



    if "coverage" in params:

        for k, color in enumerate(colors):

            # use k-th colour (sorted by presence on image)

            result["colors"][color].append({"type": "min", "k": k})

            # use k-th colour (sorted by presence on image)

            result["colors"][color].append({"type": "max", "k": len(colors) - k - 1})



    if "unique" in params:

        add_unique_colors(image, result, colors=None)

        add_center_color(image, result)



    if "corners" in params:

        # colors in the corners of images

        result["colors"][image[0, 0]].append({"type": "corner", "side": "tl"})

        result["colors"][image[0, -1]].append({"type": "corner", "side": "tr"})

        result["colors"][image[-1, 0]].append({"type": "corner", "side": "bl"})

        result["colors"][image[-1, -1]].append({"type": "corner", "side": "br"})



    if "top" in params:

        # colors that are on top of other and have full vertical on horizontal line

        for k in range(10):

            mask = image == k

            is_on_top0 = mask.min(axis=0).any()

            is_on_top1 = mask.min(axis=1).any()

            if is_on_top0:

                result["colors"][k].append({"type": "on_top", "side": "0"})

            if is_on_top1:

                result["colors"][k].append({"type": "on_top", "side": "1"})

            if is_on_top1 or is_on_top0:

                result["colors"][k].append({"type": "on_top", "side": "any"})



    if "grid" in params:

        grid_color, grid_size, frame = find_grid(image)

        if grid_color >= 0:

            result["grid_color"] = grid_color

            result["grid_size"] = grid_size

            result["grid_frame"] = frame

            result["colors"][grid_color].append({"type": "grid"})



    return result





def add_block(target_dict, image, params_list):

    array_hash = hash(matrix2answer(image))

    if array_hash not in target_dict["arrays"]:

        target_dict["arrays"][array_hash] = {"array": image, "params": []}



    for params in params_list:

        params_hash = get_dict_hash(params)

        target_dict["arrays"][array_hash]["params"].append(params)

        target_dict["params"][params_hash] = array_hash





def get_original(image):

    return 0, image





def get_inversed_colors(image):

    unique = np.unique(image)

    if len(unique) != 2:

        return 1, None

    result = image.copy()

    result[image == unique[0]] = unique[1]

    result[image == unique[1]] = unique[0]

    return 0, result





def generate_blocks(image, result, max_time=600, max_blocks=200000, max_masks=200000, target_image=None, params=None):

    all_params = [

        "initial",

        "background",

        "min_max_blocks",

        "block_with_side_colors",

        "max_area_covered",

        "grid_cells",

        "halves",

        "corners",

        "rotate",

        "transpose",

        "cut_edges",

        "resize",

        "reflect",

        "cut_parts",

        "swap_colors",

        "k_part",

    ]



    if not params:

        params = all_params



    start_time = time.time()



    result["blocks"] = {"arrays": {}, "params": {}}



    if "initial" in params:

        # starting with the original image

        add_block(result["blocks"], image, [[{"type": "original"}]])



        # inverse colors

        status, block = get_inversed_colors(image)

        if status == 0 and block.shape[0] > 0 and block.shape[1] > 0:

            add_block(result["blocks"], block, [[{"type": "inversed_colors"}]])



    # adding min and max blocks

    if (

        ("min_max_blocks" in params)

        and (time.time() - start_time < max_time)

        and (len(result["blocks"]["arrays"]) < max_blocks)

    ):

        # print("min_max_blocks")

        for full in [True, False]:

            status, block = get_max_block(image, full)

            if status == 0 and block.shape[0] > 0 and block.shape[1] > 0:

                add_block(result["blocks"], block, [[{"type": "max_block", "full": full}]])



    if (

        ("block_with_side_colors" in params)

        and (time.time() - start_time < max_time)

        and (len(result["blocks"]["arrays"]) < max_blocks)

    ):

        # print("min_max_blocks")

        for block_type in ["min", "max"]:

            for structure in [0, 1]:

                status, block = get_block_with_side_colors(image, block_type, structure)

                if status == 0 and block.shape[0] > 0 and block.shape[1] > 0:

                    add_block(

                        result["blocks"],

                        block,

                        [[{"type": "block_with_side_colors", "block_type": block_type, "structure": structure}]],

                    )

        for block_type in ["min", "max"]:

            for structure in [0, 1]:

                status, block = get_block_with_side_colors_count(image, block_type, structure)

                if status == 0 and block.shape[0] > 0 and block.shape[1] > 0:

                    add_block(

                        result["blocks"],

                        block,

                        [[{"type": "block_with_side_colors_count", "block_type": block_type, "structure": structure}]],

                    )

    # print(sum([len(x['params']) for x in result['blocks']['arrays'].values()]))

    # adding background

    if ("background" in params) and (time.time() - start_time < max_time):

        # print("background")

        for color in range(10):

            status, block = get_background(image, color)

            if status == 0 and block.shape[0] > 0 and block.shape[1] > 0:

                params_list = []

                for color_dict in result["colors"][color].copy():

                    params_list.append([{"type": "background", "color": color_dict}])

                add_block(result["blocks"], block, params_list)



    # adding the max area covered by each color

    if ("max_area_covered" in params) and (time.time() - start_time < max_time):

        # print("max_area_covered")

        for color in result["colors_sorted"]:

            status, block = get_color_max(image, color)

            if status == 0 and block.shape[0] > 0 and block.shape[1] > 0:

                params_list = []

                for color_dict in result["colors"][color].copy():

                    params_list.append([{"type": "color_max", "color": color_dict}])

                add_block(result["blocks"], block, params_list)



    # adding grid cells

    if (

        ("grid_cells" in params)

        and (time.time() - start_time < max_time)

        and (len(result["blocks"]["arrays"]) < max_blocks)

    ):

        if result["grid_color"] > 0:

            for i in range(result["grid_size"][0]):

                for j in range(result["grid_size"][1]):

                    status, block = get_grid(image, result["grid_size"], (i, j), frame=result["grid_frame"])

                    if status == 0 and block.shape[0] > 0 and block.shape[1] > 0:

                        add_block(

                            result["blocks"],

                            block,

                            [

                                [

                                    {

                                        "type": "grid",

                                        "grid_size": result["grid_size"],

                                        "cell": [i, j],

                                        "frame": result["grid_frame"],

                                    }

                                ]

                            ],

                        )



    # adding halves of the images

    if ("halves" in params) and (time.time() - start_time < max_time) and (len(result["blocks"]["arrays"]) < max_blocks):

        for side in ["l", "r", "t", "b", "long1", "long2"]:

            status, block = get_half(image, side=side)

            if status == 0 and block.shape[0] > 0 and block.shape[1] > 0:

                add_block(result["blocks"], block, [[{"type": "half", "side": side}]])



    # extracting pixels from image

    if ("pixels" in params) and (time.time() - start_time < max_time) and (len(result["blocks"]["arrays"]) < max_blocks):

        for i in range(image.shape[0]):

            for j in range(image.shape[1]):

                status, block = get_pixel(image, i=i, j=j)

                if status == 0 and block.shape[0] > 0 and block.shape[1] > 0:

                    add_block(result["blocks"], block, [[{"type": "pixel", "i": i, "j": j}]])



    # extracting pixels from image

    if (

        ("pixel_fixed" in params)

        and (time.time() - start_time < max_time)

        and (len(result["blocks"]["arrays"]) < max_blocks)

    ):

        for i in range(10):

            status, block = get_pixel_fixed(image, i=i)

            if status == 0 and block.shape[0] > 0 and block.shape[1] > 0:

                add_block(result["blocks"], block, [[{"type": "pixel_fixed", "i": i}]])



    # adding halves of the images

    if ("k_part" in params) and (time.time() - start_time < max_time) and (len(result["blocks"]["arrays"]) < max_blocks):

        for num in [3, 4]:

            for k in range(num):

                status, block = get_k_part(image, num=num, k=k)

                if status == 0 and block.shape[0] > 0 and block.shape[1] > 0:

                    add_block(result["blocks"], block, [[{"type": "k_part", "num": num, "k": k}]])



    # adding corners of the images

    if (

        ("corners" in params)

        and (time.time() - start_time < max_time)

        and (len(result["blocks"]["arrays"]) < max_blocks)

    ):

        for side in ["tl", "tr", "bl", "br"]:

            status, block = get_corner(image, side=side)

            if status == 0 and block.shape[0] > 0 and block.shape[1] > 0:

                add_block(result["blocks"], block, [[{"type": "corner", "side": side}]])



    main_blocks_num = len(result["blocks"])



    # rotate all blocks

    if ("rotate" in params) and (time.time() - start_time < max_time) and (len(result["blocks"]["arrays"]) < max_blocks):

        current_blocks = result["blocks"]["arrays"].copy()

        for k in range(1, 4):

            for key, data in current_blocks.items():

                status, block = get_rotation(data["array"], k=k)

                if status == 0 and block.shape[0] > 0 and block.shape[1] > 0:

                    params_list = [i + [{"type": "rotation", "k": k}] for i in data["params"]]

                    add_block(result["blocks"], block, params_list)



    # transpose all blocks

    if (

        ("transpose" in params)

        and (time.time() - start_time < max_time)

        and (len(result["blocks"]["arrays"]) < max_blocks)

    ):

        current_blocks = result["blocks"]["arrays"].copy()

        for key, data in current_blocks.items():

            status, block = get_transpose(data["array"])

            if status == 0 and block.shape[0] > 0 and block.shape[1] > 0:

                params_list = [i + [{"type": "transpose"}] for i in data["params"]]

                add_block(result["blocks"], block, params_list)



    # cut edges for all blocks

    if (

        ("cut_edges" in params)

        and (time.time() - start_time < max_time)

        and (len(result["blocks"]["arrays"]) < max_blocks)

    ):

        current_blocks = result["blocks"]["arrays"].copy()

        for l, r, t, b in [

            (1, 1, 1, 1),

            (1, 0, 0, 0),

            (0, 1, 0, 0),

            (0, 0, 1, 0),

            (0, 0, 0, 1),

            (1, 1, 0, 0),

            (1, 0, 0, 1),

            (0, 0, 1, 1),

            (0, 1, 1, 0),

        ]:

            if time.time() - start_time < max_time:

                for key, data in current_blocks.items():

                    status, block = get_cut_edge(data["array"], l=l, r=r, t=t, b=b)

                    if status == 0 and block.shape[0] > 0 and block.shape[1] > 0:

                        params_list = [

                            i + [{"type": "cut_edge", "l": l, "r": r, "t": t, "b": b}] for i in data["params"]

                        ]

                        add_block(result["blocks"], block, params_list)



    # resize all blocks

    if ("resize" in params) and (time.time() - start_time < max_time) and (len(result["blocks"]["arrays"]) < max_blocks):

        current_blocks = result["blocks"]["arrays"].copy()

        for scale in [2, 3, 1 / 2, 1 / 3]:

            for key, data in current_blocks.items():

                status, block = get_resize(data["array"], scale)

                if status == 0 and block.shape[0] > 0 and block.shape[1] > 0:

                    params_list = [i + [{"type": "resize", "scale": scale}] for i in data["params"]]

                    add_block(result["blocks"], block, params_list)



        for size_x, size_y in [(2, 2), (3, 3)]:

            for key, data in current_blocks.items():

                status, block = get_resize_to(data["array"], size_x, size_y)

                if status == 0 and block.shape[0] > 0 and block.shape[1] > 0:

                    params_list = [

                        i + [{"type": "resize_to", "size_x": size_x, "size_y": size_y}] for i in data["params"]

                    ]

                    add_block(result["blocks"], block, params_list)



    # reflect all blocks

    if (

        ("reflect" in params)

        and (time.time() - start_time < max_time)

        and (len(result["blocks"]["arrays"]) < max_blocks)

    ):

        current_blocks = result["blocks"]["arrays"].copy()

        for side in ["r", "l", "t", "b", "rt", "rb", "lt", "lb"]:

            if time.time() - start_time < max_time:

                for key, data in current_blocks.items():

                    status, block = get_reflect(data["array"], side)

                    if status == 0 and block.shape[0] > 0 and block.shape[1] > 0:

                        params_list = [i + [{"type": "reflect", "side": side}] for i in data["params"]]

                        add_block(result["blocks"], block, params_list)



    # cut some parts of images

    if (

        ("cut_parts" in params)

        and (time.time() - start_time < max_time)

        and (len(result["blocks"]["arrays"]) < max_blocks)

    ):

        max_x = image.shape[0]

        max_y = image.shape[1]

        min_block_size = 2

        for x1 in range(0, max_x - min_block_size):

            if time.time() - start_time < max_time:

                if max_x - x1 <= min_block_size:

                    continue

                for x2 in range(x1 + min_block_size, max_x):

                    for y1 in range(0, max_y - min_block_size):

                        if max_y - y1 <= min_block_size:

                            continue

                        for y2 in range(y1 + min_block_size, max_y):

                            status, block = get_cut(image, x1, y1, x2, y2)

                            if status == 0:

                                add_block(

                                    result["blocks"], block, [[{"type": "cut", "x1": x1, "x2": x2, "y1": y1, "y2": y2}]]

                                )



    list_param_list = []

    list_blocks = []



    # swap some colors

    if (

        ("swap_colors" in params)

        and (time.time() - start_time < max_time)

        and (len(result["blocks"]["arrays"]) < max_blocks)

    ):

        current_blocks = result["blocks"]["arrays"].copy()

        for color_1 in range(9):

            if time.time() - start_time < max_time:

                for color_2 in range(color_1 + 1, 10):

                    for key, data in current_blocks.items():

                        status, block = get_color_swap(data["array"], color_1, color_2)

                        if status == 0 and block.shape[0] > 0 and block.shape[1] > 0:

                            for color_dict_1 in result["colors"][color_1].copy():

                                for color_dict_2 in result["colors"][color_2].copy():

                                    list_param_list.append(

                                        [

                                            j

                                            + [{"type": "color_swap", "color_1": color_dict_1, "color_2": color_dict_2}]

                                            for j in data["params"]

                                        ]

                                    )

                                    list_blocks.append(block)



    for block, params_list in zip(list_blocks, list_param_list):

        add_block(result["blocks"], block, params_list)



    if time.time() - start_time > max_time:

        print("Time is over")

    if len(result["blocks"]["arrays"]) >= max_blocks:

        print("Max number of blocks exceeded")

    return result





def generate_masks(image, result, max_time=600, max_blocks=200000, max_masks=200000, target_image=None, params=None):

    start_time = time.time()



    all_params = ["initial_masks", "additional_masks", "coverage_masks", "min_max_masks"]



    if not params:

        params = all_params



    result["masks"] = {"arrays": {}, "params": {}}



    # making one mask for each generated block

    current_blocks = result["blocks"]["arrays"].copy()

    if ("initial_masks" in params) and (time.time() - start_time < max_time * 2):

        for key, data in current_blocks.items():

            for color in result["colors_sorted"]:

                status, mask = get_mask_from_block(data["array"], color)

                if status == 0 and mask.shape[0] > 0 and mask.shape[1] > 0:

                    params_list = [

                        {"operation": "none", "params": {"block": i, "color": color_dict}}

                        for i in data["params"]

                        for color_dict in result["colors"][color]

                    ]

                    add_block(result["masks"], mask, params_list)



    initial_masks = result["masks"]["arrays"].copy()

    if ("initial_masks" in params) and (time.time() - start_time < max_time * 2):

        for key, mask in initial_masks.items():

            add_block(

                result["masks"],

                np.logical_not(mask["array"]),

                [{"operation": "not", "params": param["params"]} for param in mask["params"]],

            )



    initial_masks = result["masks"]["arrays"].copy()

    masks_to_add = []

    processed = []

    if ("additional_masks" in params) and (time.time() - start_time < max_time * 2):

        for key1, mask1 in initial_masks.items():

            processed.append(key1)

            if time.time() - start_time < max_time * 2 and (

                target_image is None

                or (target_image.shape == mask1["array"].shape)

                or (target_image.shape == mask1["array"].T.shape)

            ):

                for key2, mask2 in initial_masks.items():

                    if key2 in processed:

                        continue

                    if (mask1["array"].shape[0] == mask2["array"].shape[0]) and (

                        mask1["array"].shape[1] == mask2["array"].shape[1]

                    ):

                        params_list_and = []

                        params_list_or = []

                        params_list_xor = []

                        for param1 in mask1["params"]:

                            for param2 in mask2["params"]:

                                params_list_and.append(

                                    {"operation": "and", "params": {"mask1": param1, "mask2": param2}}

                                )

                                params_list_or.append({"operation": "or", "params": {"mask1": param1, "mask2": param2}})

                                params_list_xor.append(

                                    {"operation": "xor", "params": {"mask1": param1, "mask2": param2}}

                                )

                        masks_to_add.append(

                            (result["masks"], np.logical_and(mask1["array"], mask2["array"]), params_list_and)

                        )

                        masks_to_add.append(

                            (result["masks"], np.logical_or(mask1["array"], mask2["array"]), params_list_or)

                        )

                        masks_to_add.append(

                            (result["masks"], np.logical_xor(mask1["array"], mask2["array"]), params_list_xor)

                        )



    for path, array, params_list in masks_to_add:

        add_block(path, array, params_list)

    # coverage_masks

    if ("coverage_masks" in params) and (time.time() - start_time < max_time * 2):

        for color in result["colors_sorted"][1:]:

            status, mask = get_mask_from_max_color_coverage(image, color)

            if status == 0 and mask.shape[0] > 0 and mask.shape[1] > 0:

                params_list = [

                    {"operation": "coverage", "params": {"color": color_dict}}

                    for color_dict in result["colors"][color].copy()

                ]

                add_block(result["masks"], mask, params_list)

    # coverage_masks

    if ("min_max_masks" in params) and (time.time() - start_time < max_time * 2):

        status, mask = get_min_block_mask(image)

        if status == 0 and mask.shape[0] > 0 and mask.shape[1] > 0:

            params_list = [{"operation": "min_block"}]

            add_block(result["masks"], mask, params_list)

        status, mask = get_max_block_mask(image)

        if status == 0 and mask.shape[0] > 0 and mask.shape[1] > 0:

            params_list = [{"operation": "max_block"}]

            add_block(result["masks"], mask, params_list)

    if time.time() - start_time > max_time:

        print("Time is over")

    if len(result["blocks"]["arrays"]) >= max_masks:

        print("Max number of masks exceeded")

    return result





def process_image(

    image, max_time=600, max_blocks=200000, max_masks=200000, target_image=None, params=None, color_params=None

):

    """processes the original image and returns dict with structured image blocks"""



    result = get_color_scheme(image, target_image=target_image, params=color_params)

    result = generate_blocks(image, result, max_time, max_blocks, max_masks, target_image, params, color_params)

    result = generate_masks(image, result, max_time, max_blocks, max_masks, target_image, params, color_params)



    return result





def get_mask_from_block_params(image, params, block_cache=None, mask_cache=None, color_scheme=None):

    if mask_cache is None:

        mask_cache = {{"arrays": {}, "params": {}}}

    dict_hash = get_dict_hash(params)

    if dict_hash in mask_cache:

        mask = mask_cache["arrays"][mask_cache["params"][dict_hash]]["array"]

        if len(mask) == 0:

            return 1, None

        else:

            return 0, mask



    if params["operation"] == "none":

        status, block = get_predict(image, params["params"]["block"], block_cache, color_scheme)

        if status != 0:

            add_block(mask_cache, np.array([[]]), [params])

            return 1, None

        if not color_scheme:

            color_scheme = get_color_scheme(image)

        color_num = get_color(params["params"]["color"], color_scheme["colors"])

        if color_num < 0:

            add_block(mask_cache, np.array([[]]), [params])

            return 2, None

        status, mask = get_mask_from_block(block, color_num)

        if status != 0:

            add_block(mask_cache, np.array([[]]), [params])

            return 6, None

        add_block(mask_cache, mask, [params])

        return 0, mask

    elif params["operation"] == "not":

        new_params = params.copy()

        new_params["operation"] = "none"

        status, mask = get_mask_from_block_params(

            image, new_params, block_cache=block_cache, color_scheme=color_scheme, mask_cache=mask_cache

        )

        if status != 0:

            add_block(mask_cache, np.array([[]]), [params])

            return 3, None

        mask = np.logical_not(mask)

        add_block(mask_cache, mask, [params])

        return 0, mask

    elif params["operation"] in ["and", "or", "xor"]:

        new_params = params["params"]["mask1"]

        status, mask1 = get_mask_from_block_params(

            image, new_params, block_cache=block_cache, color_scheme=color_scheme, mask_cache=mask_cache

        )

        if status != 0:

            add_block(mask_cache, np.array([[]]), [params])

            return 4, None

        new_params = params["params"]["mask2"]

        status, mask2 = get_mask_from_block_params(

            image, new_params, block_cache=block_cache, color_scheme=color_scheme, mask_cache=mask_cache

        )

        if status != 0:

            add_block(mask_cache, np.array([[]]), [params])

            return 5, None

        if mask1.shape[0] != mask2.shape[0] or mask1.shape[1] != mask2.shape[1]:

            add_block(mask_cache, np.array([[]]), [params])

            return 6, None

        if params["operation"] == "and":

            mask = np.logical_and(mask1, mask2)

        elif params["operation"] == "or":

            mask = np.logical_or(mask1, mask2)

        elif params["operation"] == "xor":

            mask = np.logical_xor(mask1, mask2)

        add_block(mask_cache, mask, [params])

        return 0, mask

    elif params["operation"] == "coverage":

        if not color_scheme:

            color_scheme = get_color_scheme(image)

        color_num = get_color(params["params"]["color"], color_scheme["colors"])

        if color_num < 0:

            add_block(mask_cache, np.array([[]]), [params])

            return 2, None

        status, mask = get_mask_from_max_color_coverage(image, color_num)

        if status != 0:

            add_block(mask_cache, np.array([[]]), [params])

            return 6, None

        add_block(mask_cache, mask, [params])

        return 0, mask

    elif params["operation"] == "min_block":

        status, mask = get_min_block_mask(image)

        if status != 0:

            add_block(mask_cache, np.array([[]]), [params])

            return 6, None

        add_block(mask_cache, mask, [params])

        return 0, mask

    elif params["operation"] == "max_block":

        status, mask = get_max_block_mask(image)

        if status != 0:

            add_block(mask_cache, np.array([[]]), [params])

            return 6, None

        add_block(mask_cache, mask, [params])

        return 0, mask





def get_dict_hash(d):

    return hash(json.dumps(d, sort_keys=True))





def get_predict(image, transforms, block_cache=None, color_scheme=None):

    """ applies the list of transforms to the image"""

    params_hash = get_dict_hash(transforms)

    if params_hash in block_cache["params"]:

        if block_cache["params"][params_hash] is None:

            return 1, None

        else:

            return 0, block_cache["arrays"][block_cache["params"][params_hash]]["array"]



    if not color_scheme:

        color_scheme = get_color_scheme(image)



    if len(transforms) > 1:

        status, previous_image = get_predict(image, transforms[:-1], block_cache=block_cache, color_scheme=color_scheme)

        if status != 0:

            return status, None

    else:

        previous_image = image



    transform = transforms[-1]

    function = globals()["get_" + transform["type"]]

    params = transform.copy()

    params.pop("type")

    for color_name in ["color", "color_1", "color_2"]:

        if color_name in params:

            params[color_name] = get_color(params[color_name], color_scheme["colors"])

            if params[color_name] < 0:

                return 2, None

    status, result = function(previous_image, **params)



    if status != 0 or len(result) == 0 or len(result[0]) == 0:

        block_cache["params"][params_hash] = None

        return 1, None



    add_block(block_cache, result, [transforms])

    return 0, result





def filter_colors(sample):

    # filtering colors, that are not present in at least one of the images

    all_colors = []

    for color_scheme1 in sample["train"]:

        list_of_colors = [get_dict_hash(color_dict) for i in range(10) for color_dict in color_scheme1["colors"][i]]

        all_colors.append(list_of_colors)

    for j in range(1, len(sample["train"])):

        all_colors[0] = [x for x in all_colors[0] if x in all_colors[j]]

    keep_colors = set(all_colors[0])



    for color_scheme1 in sample["train"]:

        for i in range(10):

            j = 0

            while j < len(color_scheme1["colors"][i]):

                if get_dict_hash(color_scheme1["colors"][i][j]) in keep_colors:

                    j += 1

                else:

                    del color_scheme1["colors"][i][j]



    delete_colors = []

    color_scheme0 = sample["train"][0]

    for i in range(10):

        if len(color_scheme0["colors"][i]) > 1:

            for j, color_dict1 in enumerate(color_scheme0["colors"][i][::-1][:-1]):

                hash1 = get_dict_hash(color_dict1)

                delete = True

                for color_dict2 in color_scheme0["colors"][i][::-1][j + 1 :]:

                    hash2 = get_dict_hash(color_dict2)

                    for color_scheme1 in list(sample["train"][1:]) + list(sample["test"]):

                        found = False

                        for k in range(10):

                            hash_array = [get_dict_hash(color_dict) for color_dict in color_scheme1["colors"][k]]

                            if hash1 in hash_array and hash2 in hash_array:

                                found = True

                                break

                        if not found:

                            delete = False

                            break

                    if delete:

                        delete_colors.append(hash1)

                        break



    for color_scheme1 in sample["train"]:

        for i in range(10):

            j = 0

            while j < len(color_scheme1["colors"][i]):

                if get_dict_hash(color_scheme1["colors"][i][j]) in delete_colors:

                    del color_scheme1["colors"][i][j]

                else:

                    j += 1

    return





def filter_blocks(sample, arrays_type="blocks"):

    delete_blocks = []

    list_of_lists_of_sets = []

    for arrays_list in [x[arrays_type]["arrays"].values() for x in sample["train"][1:]] + [

        x[arrays_type]["arrays"].values() for x in sample["test"]

    ]:

        list_of_lists_of_sets.append([])

        for array in arrays_list:

            list_of_lists_of_sets[-1].append({get_dict_hash(params_dict) for params_dict in array["params"]})



    for initial_array in sample["train"][0][arrays_type]["arrays"].values():

        if len(initial_array["params"]) > 1:

            for j, params_dict1 in enumerate(initial_array["params"][::-1][:-1]):

                hash1 = get_dict_hash(params_dict1)

                delete = True

                for params_dict1 in initial_array["params"][::-1][j + 1 :]:

                    hash2 = get_dict_hash(params_dict1)

                    for lists_of_sets in list_of_lists_of_sets:

                        found = False

                        for hash_set in lists_of_sets:

                            if hash1 in hash_set and hash2 in hash_set:

                                found = True

                                break

                        if not found:

                            delete = False

                            break

                    if delete:

                        delete_blocks.append(hash1)

                        break



    for arrays_list in [x[arrays_type]["arrays"].values() for x in sample["train"]] + [

        x[arrays_type]["arrays"].values() for x in sample["test"]

    ]:

        for array in arrays_list:

            params_list = array["params"]

            j = 0

            while j < len(params_list):

                if get_dict_hash(params_list[j]) in delete_blocks:

                    del params_list[j]

                else:

                    j += 1

    return





def extract_target_blocks(sample, color_params=None):

    target_blocks_cache = []

    params = ["initial", "block_with_side_colors", "min_max_blocks", "max_area_covered", "cut_parts"]

    for n in range(len(sample["train"])):

        target_image = np.uint8(sample["train"][n]["output"])

        target_blocks_cache.append(get_color_scheme(target_image, params=color_params))

        target_blocks_cache[-1].update(generate_blocks(target_image, target_blocks_cache[-1], params=params))

    final_arrays = list(

        set.intersection(

            *[set(target_blocks_cache[n]["blocks"]["arrays"].keys()) for n in range(len(target_blocks_cache))]

        )

    )

    for i, key in enumerate(final_arrays):

        for n in range(len(sample["train"])):

            params_list = [[{"type": "target", "k": i}]]

            add_block(

                sample["train"][n]["blocks"], target_blocks_cache[0]["blocks"]["arrays"][key]["array"], params_list

            )

        for n in range(len(sample["test"])):

            params_list = [[{"type": "target", "k": i}]]

            add_block(sample["test"][n]["blocks"], target_blocks_cache[0]["blocks"]["arrays"][key]["array"], params_list)





def preprocess_sample(sample, params=None, color_params=None, process_whole_ds=False):

    """ make the whole preprocessing for particular sample"""



    for n, image in enumerate(sample["train"]):

        original_image = np.uint8(image["input"])

        target_image = np.uint8(sample["train"][n]["output"])

        sample["train"][n].update(get_color_scheme(original_image, target_image=target_image, params=color_params))

    for n, image in enumerate(sample["test"]):

        original_image = np.uint8(image["input"])

        sample["test"][n].update(get_color_scheme(original_image, params=color_params))



    filter_colors(sample)



    for n, image in enumerate(sample["train"]):

        original_image = np.uint8(image["input"])

        target_image = np.uint8(sample["train"][n]["output"])

        sample["train"][n].update(

            generate_blocks(original_image, sample["train"][n], target_image=target_image, params=params)

        )

    for n, image in enumerate(sample["test"]):

        original_image = np.uint8(image["input"])

        sample["test"][n].update(generate_blocks(original_image, sample["test"][n], params=params))



    if "target" in params:

        extract_target_blocks(sample, color_params)

    filter_blocks(sample)



    for n, image in enumerate(sample["train"]):

        original_image = np.uint8(image["input"])

        target_image = np.uint8(sample["train"][n]["output"])

        sample["train"][n].update(

            generate_masks(original_image, sample["train"][n], target_image=target_image, params=params)

        )

    for n, image in enumerate(sample["test"]):

        original_image = np.uint8(image["input"])

        sample["test"][n].update(generate_masks(original_image, sample["test"][n], params=params))



    return sample

import numpy as np





def filter_list_of_dicts(list1, list2):

    """Returns the intersection of two lists of dicts"""

    set_of_hashes = {get_dict_hash(item1) for item1 in list1}

    final_list = []

    for item2 in list2:

        if get_dict_hash(item2) in set_of_hashes:

            final_list.append(item2)

    return final_list





def swap_two_colors(image):

    """sawaps two colors"""

    unique = np.unique(image)

    if len(unique) != 2:

        return 1, None

    result = image.copy()

    result[image == unique[0]] = unique[1]

    result[image == unique[1]] = unique[0]

    return 0, result





def combine_two_lists(list1, list2):

    result = list1.copy()

    for item2 in list2:

        exist = False

        for item1 in list1:

            if (item2 == item1).all():

                exist = True

                break

        if not exist:

            result.append(item2)

    return result





def intersect_two_lists(list1, list2):

    """ intersects two lists of np.arrays"""

    result = []

    for item2 in list2:

        for item1 in list1:

            if (item2.shape == item1.shape) and (item2 == item1).all():

                result.append(item2)

                break

    return result
import itertools

import random



import numpy as np



from scipy import ndimage

from scipy.stats import mode



class Predictor:

    def __init__(self, params=None, preprocess_params=None):

        if params is None:

            self.params = {}

        else:

            self.params = params

        self.preprocess_params = preprocess_params

        self.solution_candidates = []

        if "rrr_input" in self.params:

            self.rrr_input = params["rrr_input"]

        else:

            self.rrr_input = True

        if "mosaic_target" not in self.params:

            self.params["mosaic_target"] = False



    def retrive_params_values(self, params, color_scheme):

        new_params = {}

        for k, v in params.items():

            if k[-5:] == "color":

                new_params[k] = get_color(v, color_scheme["colors"])

                if new_params[k] < 0:

                    return 1, None

            else:

                new_params[k] = v

        return 0, new_params



    def reflect_rotate_roll(self, image, inverse=False):

        if self.params is not None and "reflect" in self.params:

            reflect = self.params["reflect"]

        else:

            reflect = (False, False)

        if self.params is not None and "rotate" in self.params:

            rotate = self.params["rotate"]

        else:

            rotate = 0

        if self.params is not None and "roll" in self.params:

            roll = self.params["roll"]

        else:

            roll = (0, 0)



        result = image.copy()



        if inverse:

            if reflect[0]:

                result = result[::-1]

            if reflect[1]:

                result = result[:, ::-1]

            result = np.rot90(result, -rotate)

            result = np.roll(result, -roll[1], axis=1)

            result = np.roll(result, -roll[0], axis=0)

        else:

            result = np.roll(result, roll[0], axis=0)

            result = np.roll(result, roll[1], axis=1)

            result = np.rot90(result, rotate)

            if reflect[1]:

                result = result[:, ::-1]

            if reflect[0]:

                result = result[::-1]



        return result



    def get_images(self, k, train=True, return_target=True):

        if not train:

            return_target = False



        if train:

            if self.rrr_input:

                original_image = self.reflect_rotate_roll(np.uint8(self.sample["train"][k]["input"]))

            else:

                original_image = np.uint8(self.sample["train"][k]["input"])

            if return_target:

                if self.params["mosaic_target"]:

                    target_image = np.uint8(self.sample["train"][k]["mosaic_output"])

                else:

                    target_image = np.uint8(self.sample["train"][k]["output"])

                target_image = self.reflect_rotate_roll(target_image)

                return original_image, target_image

            else:

                return original_image

        else:

            if self.rrr_input:

                original_image = self.reflect_rotate_roll(np.uint8(self.sample["test"][k]["input"]))

            else:

                original_image = np.uint8(self.sample["test"][k]["input"])

            return original_image



    def initiate_mosaic(self):

        same_size = True

        same_size_rotated = True

        fixed_size = True

        color_num_size = True

        block_shape_size = True



        shapes = []

        sizes = []

        for k, data in enumerate(self.sample["train"]):

            target_image = np.uint8(data["output"])

            original_image = self.get_images(k, train=True, return_target=False)

            status, block = find_mosaic_block(target_image, self.params)

            if status != 0:

                return False

            self.sample["train"][k]["mosaic_output"] = block

            same_size = same_size and target_image.shape == original_image.shape

            same_size_rotated = same_size_rotated and target_image.shape == original_image.T.shape

            if target_image.shape[0] % block.shape[0] == 0 and target_image.shape[1] % block.shape[1] == 0:

                sizes.append([target_image.shape[0] // block.shape[0], target_image.shape[1] // block.shape[1]])

                color_num_size = (

                    color_num_size

                    and sizes[-1][0] == len(data["colors_sorted"])

                    and sizes[-1][1] == len(data["colors_sorted"])

                )

                block_shape_size = block_shape_size and sizes[-1][0] == block.shape[0] and sizes[-1][1] == block.shape[1]

            else:

                fixed_size = False

                color_num_size = False

                block_shape_size

            shapes.append(target_image.shape)



        params = {}



        if len([1 for x in shapes[1:] if x != shapes[0]]) == 0:

            params["mosaic_size_type"] = "fixed"

            params["mosaic_shape"] = shapes[0]

        elif fixed_size and len([1 for x in sizes[1:] if x != sizes[0]]) == 0:

            params["mosaic_size_type"] = "size"

            params["mosaic_size"] = sizes[0]

        elif same_size:

            params["mosaic_size_type"] = "same"

        elif same_size_rotated:

            params["mosaic_size_type"] = "same_rotated"

        elif color_num_size:

            params["mosaic_size_type"] = "color_num"

        elif color_num_size:

            params["mosaic_size_type"] = "block_shape_size"

        else:

            return False



        self.params["mosaic_params"] = params

        return True



    def process_prediction(self, image, original_image=None):

        result = self.reflect_rotate_roll(image, inverse=True)

        if self.params["mosaic_target"]:

            result = reconstruct_mosaic_from_block(result, self.params["mosaic_params"], original_image=original_image)

        return result



    def predict_output(self, image, params):

        """ predicts 1 output image given input image and prediction params"""

        return 1, None



    def filter_colors(self):

        # filtering colors, that are not present in at least one of the images

        all_colors = []

        for color_scheme1 in self.sample["train"]:

            list_of_colors = [get_dict_hash(color_dict) for i in range(10) for color_dict in color_scheme1["colors"][i]]

            all_colors.append(list_of_colors)

        for j in range(1, len(self.sample["train"])):

            all_colors[0] = [x for x in all_colors[0] if x in all_colors[j]]

        keep_colors = set(all_colors[0])



        for color_scheme1 in self.sample["train"]:

            for i in range(10):

                j = 0

                while j < len(color_scheme1["colors"][i]):

                    if get_dict_hash(color_scheme1["colors"][i][j]) in keep_colors:

                        j += 1

                    else:

                        del color_scheme1["colors"][i][j]



        delete_colors = []

        color_scheme0 = self.sample["train"][0]

        for i in range(10):

            if len(color_scheme0["colors"][i]) > 1:

                for j, color_dict1 in enumerate(color_scheme0["colors"][i][::-1][:-1]):

                    hash1 = get_dict_hash(color_dict1)

                    delete = True

                    for color_dict2 in color_scheme0["colors"][i][::-1][j + 1 :]:

                        hash2 = get_dict_hash(color_dict2)

                        for color_scheme1 in list(self.sample["train"][1:]) + list(self.sample["test"]):

                            found = False

                            for k in range(10):

                                hash_array = [get_dict_hash(color_dict) for color_dict in color_scheme1["colors"][k]]

                                if hash1 in hash_array and hash2 in hash_array:

                                    found = True

                                    break

                            if not found:

                                delete = False

                                break

                        if delete:

                            delete_colors.append(hash1)

                            break



        for color_scheme1 in self.sample["train"]:

            for i in range(10):

                j = 0

                while j < len(color_scheme1["colors"][i]):

                    if get_dict_hash(color_scheme1["colors"][i][j]) in delete_colors:

                        del color_scheme1["colors"][i][j]

                    else:

                        j += 1

        return



    def filter_sizes(self):

        if "max_size" not in self.params:

            return True

        else:

            max_size = self.params["max_size"]

        for n in range(len(self.sample["train"])):

            original_image = np.array(self.sample["train"][n]["input"])

            target_image = np.array(self.sample["train"][n]["output"])

            if (

                original_image.shape[0] > max_size

                or original_image.shape[1] > max_size

                or target_image.shape[0] > max_size

                or target_image.shape[1] > max_size

            ):

                return False

        return True



    def init_call(self):

        if not self.filter_sizes():

            return False

        self.filter_colors()

        if self.params["mosaic_target"]:

            if self.initiate_mosaic():

                return True

            else:

                return False

        return True



    def process_one_sample(self, k, initial=False):

        """ processes k train sample and updates self.solution_candidates"""

        return 0



    def process_full_train(self):

        for k in range(len(self.sample["train"])):

            status = self.process_one_sample(k, initial=(k == 0))

            if status != 0:

                return 1



        if len(self.solution_candidates) == 0:

            return 2



        return 0



    def add_candidates_list(self, image, target_image, color_scheme, params):

        old_params = params.copy()

        params = params.copy()

        params["color_scheme"] = color_scheme

        params["block_cache"] = color_scheme["blocks"]

        params["mask_cache"] = color_scheme["masks"]



        if "elim_background" in self.params and self.params["elim_background"]:

            structure = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]



            if "all_background_color" in params:

                color_iter_list = [params["all_background_color"]]

            else:

                color_iter_list = color_scheme["colors_sorted"]

            for all_background_color in color_iter_list:

                final_prediction = image.copy()

                solved = True

                masks, n_masks = ndimage.label(image != all_background_color, structure=structure)

                new_image_masks = [(masks == i) for i in range(1, n_masks + 1)]

                for image_mask in new_image_masks:

                    boundaries = find_color_boundaries(image_mask, True)

                    new_image = image[boundaries[0] : boundaries[1] + 1, boundaries[2] : boundaries[3] + 1]

                    new_target = target_image[boundaries[0] : boundaries[1] + 1, boundaries[2] : boundaries[3] + 1]

                    if "block" in params:

                        status, prediction = self.predict_output(new_image, params, block=new_image)

                    else:

                        status, prediction = self.predict_output(new_image, params)

                    if status != 0 or prediction.shape != new_target.shape or not (prediction == new_target).all():

                        solved = False

                        break

                    final_prediction[boundaries[0] : boundaries[1] + 1, boundaries[2] : boundaries[3] + 1] = prediction

                if solved and final_prediction.shape == target_image.shape and (final_prediction == target_image).all():

                    params["all_background_color"] = all_background_color

                    break

                else:

                    solved = False

            if not solved:

                return []



        else:

            status, prediction = self.predict_output(image, params)

            if status != 0 or prediction.shape != target_image.shape or not (prediction == target_image).all():

                return []



        result = [old_params.copy()]

        for k, v in params.copy().items():

            if k[-5:] == "color":

                temp_result = result.copy()

                result = []

                for dict in temp_result:

                    for color_dict in color_scheme["colors"][v]:

                        temp_dict = dict.copy()

                        temp_dict[k] = color_dict

                        result.append(temp_dict)



        return result



    def update_solution_candidates(self, local_candidates, initial):

        if initial:

            self.solution_candidates = local_candidates

        else:

            self.solution_candidates = filter_list_of_dicts(local_candidates, self.solution_candidates)

        if len(self.solution_candidates) == 0:

            return 4

        else:

            return 0



    def __call__(self, sample):

        """ works like fit_predict"""

        self.sample = sample



        self.initial_train = list(sample["train"]).copy()



        if self.params is not None and "skip_train" in self.params:

            skip_train = min(len(sample["train"]) - 2, self.params["skip_train"])

            train_len = len(self.initial_train) - skip_train

        else:

            train_len = len(self.initial_train)



        answers = []

        for _ in self.sample["test"]:

            answers.append([])

        result_generated = False



        all_subsets = list(itertools.combinations(self.initial_train, train_len))

        for subset in all_subsets:

            self.sample["train"] = subset

            if not self.init_call():

                continue

            status = self.process_full_train()

            if status != 0:

                continue



            for test_n, test_data in enumerate(self.sample["test"]):

                original_image = self.get_images(test_n, train=False)

                color_scheme = self.sample["test"][test_n]

                for params_dict in self.solution_candidates:

                    status, params = self.retrive_params_values(params_dict, color_scheme)

                    if status != 0:

                        continue

                    params["block_cache"] = self.sample["test"][test_n]["blocks"]

                    params["mask_cache"] = self.sample["test"][test_n]["masks"]

                    params["color_scheme"] = self.sample["test"][test_n]

                    status, prediction = self.predict_output(original_image, params)

                    if status != 0:

                        continue



                    if "elim_background" in self.params and self.params["elim_background"]:

                        result = original_image.copy()

                        structure = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]



                        all_background_color = params["all_background_color"]

                        solved = True

                        masks, n_masks = ndimage.label(original_image != all_background_color, structure=structure)

                        new_image_masks = [(masks == i) for i in range(1, n_masks + 1)]

                        for image_mask in new_image_masks:

                            boundaries = find_color_boundaries(image_mask, True)

                            new_image = original_image[

                                boundaries[0] : boundaries[1] + 1, boundaries[2] : boundaries[3] + 1

                            ]

                            if "block" in params:

                                status, prediction = self.predict_output(new_image, params, block=new_image)

                            else:

                                status, prediction = self.predict_output(new_image, params)

                            if status != 0 or prediction.shape != new_image.shape:

                                solved = False

                                break

                            result[boundaries[0] : boundaries[1] + 1, boundaries[2] : boundaries[3] + 1] = prediction

                        if not solved:

                            continue

                        prediction = result



                    else:

                        status, prediction = self.predict_output(original_image, params)

                        if status != 0:

                            continue



                    answers[test_n].append(self.process_prediction(prediction, original_image=original_image))

                    result_generated = True



        self.sample["train"] = self.initial_train

        if result_generated:

            return 0, answers

        else:

            return 3, None

        

class Puzzle(Predictor):

    """Stack different blocks together to get the output"""



    def __init__(self, params=None, preprocess_params=None):

        super().__init__(params, preprocess_params)

        self.intersection = params["intersection"]



    def initiate_factors(self, target_image):

        t_n, t_m = target_image.shape

        factors = []

        grid_color_list = []

        if self.intersection < 0:

            grid_color, grid_size, frame = find_grid(target_image)

            if grid_color < 0:

                return factors, []

            factors = [grid_size]

            grid_color_list = self.sample["train"][0]["colors"][grid_color]

            self.frame = frame

        else:

            for i in range(1, t_n + 1):

                for j in range(1, t_m + 1):

                    if (t_n - self.intersection) % i == 0 and (t_m - self.intersection) % j == 0:

                        factors.append([i, j])

        return factors, grid_color_list



    def predict_output(self, image, color_scheme, factor, params, block_cache):

        """ predicts 1 output image given input image and prediction params"""

        skip = False

        for i in range(factor[0]):

            for j in range(factor[1]):

                status, array = get_predict(image, params[i][j][0], block_cache, color_scheme)

                if status != 0:

                    skip = True

                    break



                if i == 0 and j == 0:

                    n, m = array.shape

                    predict = np.uint8(

                        np.zeros(

                            (

                                (n - self.intersection) * factor[0] + self.intersection,

                                (m - self.intersection) * factor[1] + self.intersection,

                            )

                        )

                    )

                    if self.intersection < 0:

                        new_grid_color = get_color(self.grid_color_list[0], color_scheme["colors"])

                        if new_grid_color < 0:

                            return 2, None

                        predict += new_grid_color

                else:

                    if n != array.shape[0] or m != array.shape[1]:

                        skip = True

                        break



                predict[

                    i * (n - self.intersection) : i * (n - self.intersection) + n,

                    j * (m - self.intersection) : j * (m - self.intersection) + m,

                ] = array



            if skip:

                return 1, None



        if self.intersection < 0 and self.frame:

            final_predict = predict = (

                np.uint8(

                    np.zeros(

                        (

                            (n - self.intersection) * factor[0] + self.intersection + 2,

                            (m - self.intersection) * factor[1] + self.intersection + 2,

                        )

                    )

                )

                + new_grid_color

            )

            final_predict[1 : final_predict.shape[0] - 1, 1 : final_predict.shape[1] - 1] = predict

            preict = final_predict



        return 0, predict



    def initiate_candidates_list(self, initial_values=None):

        """creates an empty candidates list corresponding to factors

        for each (m,n) factor it is m x n matrix of lists"""

        candidates = []

        if not initial_values:

            initial_values = []

        for n_factor, factor in enumerate(self.factors):

            candidates.append([])

            for i in range(factor[0]):

                candidates[n_factor].append([])

                for j in range(factor[1]):

                    candidates[n_factor][i].append(initial_values.copy())

        return candidates



    def process_one_sample(self, k, initial=False):

        """ processes k train sample and updates self.solution_candidates"""



        original_image, target_image = self.get_images(k)



        candidates_num = 0

        t_n, t_m = target_image.shape

        color_scheme = self.sample["train"][k]

        new_candidates = self.initiate_candidates_list()

        for n_factor, factor in enumerate(self.factors.copy()):

            for i in range(factor[0]):

                for j in range(factor[1]):

                    if initial:

                        local_candidates = self.sample["train"][k]["blocks"]["arrays"].keys()

                    else:

                        local_candidates = self.solution_candidates[n_factor][i][j]



                    for data in local_candidates:

                        if initial:

                            array = self.sample["train"][k]["blocks"]["arrays"][data]["array"]

                            params = self.sample["train"][k]["blocks"]["arrays"][data]["params"]

                        else:

                            params = [data]

                            status, array = get_predict(

                                original_image, data, self.sample["train"][k]["blocks"], color_scheme

                            )

                            if status != 0:

                                continue



                        n, m = array.shape

                        # work with valid candidates only

                        if n <= 0 or m <= 0:

                            continue

                        if (

                            n - self.intersection != (t_n - self.intersection) / factor[0]

                            or m - self.intersection != (t_m - self.intersection) / factor[1]

                        ):

                            continue



                        start_n = i * (n - self.intersection)

                        start_m = j * (m - self.intersection)



                        if not (

                            (n == target_image[start_n : start_n + n, start_m : start_m + m].shape[0])

                            and (m == target_image[start_n : start_n + n, start_m : start_m + m].shape[1])

                        ):

                            continue



                        # adding the candidate to the candidates list

                        if (array == target_image[start_n : start_n + n, start_m : start_m + m]).all():

                            new_candidates[n_factor][i][j].extend(params)

                            candidates_num += 1

                    # if there is no candidates for one of the cells the whole factor is invalid

                    if len(new_candidates[n_factor][i][j]) == 0:

                        self.factors[n_factor] = [0, 0]

                        break

                if self.factors[n_factor][0] == 0:

                    break



        self.solution_candidates = new_candidates



        if candidates_num > 0:

            return 0

        else:

            return 1



    def filter_factors(self, local_factors):

        for factor in self.factors:

            found = False

            for new_factor in local_factors:

                if factor == new_factor:

                    found = True

                    break

            if not found:

                factor = [0, 0]



        return



    def process_full_train(self):



        for k in range(len(self.sample["train"])):

            original_image, target_image = self.get_images(k)

            if k == 0:

                self.factors, self.grid_color_list = self.initiate_factors(target_image)

            else:

                local_factors, grid_color_list = self.initiate_factors(target_image)

                self.filter_factors(local_factors)

                self.grid_color_list = filter_list_of_dicts(grid_color_list, self.grid_color_list)



            status = self.process_one_sample(k, initial=(k == 0))

            if status != 0:

                return 1



        if len(self.solution_candidates) == 0:

            return 2



        return 0



    def __call__(self, sample):

        """ works like fit_predict"""

        self.sample = sample

        if not self.init_call():

            return 5, None

        status = self.process_full_train()

        if status != 0:

            return status, None



        answers = []

        for _ in self.sample["test"]:

            answers.append([])



        result_generated = False

        for test_n, test_data in enumerate(self.sample["test"]):

            original_image = self.get_images(test_n, train=False)

            color_scheme = self.sample["test"][test_n]

            for n_factor, factor in enumerate(self.factors):

                if factor[0] > 0 and factor[1] > 0:

                    status, prediction = self.predict_output(

                        original_image,

                        color_scheme,

                        factor,

                        self.solution_candidates[n_factor],

                        self.sample["test"][test_n]["blocks"],

                    )

                    if status == 0:

                        answers[test_n].append(self.process_prediction(prediction, original_image=original_image))

                        result_generated = True



        if result_generated:

            if "mode" in self.params and self.params["mode"]:

                for i in range(len(answers)):

                    answer = mode(np.stack(answers[i]), axis=0).mode[0]

                    answers[i] = [answer]

            return 0, answers

        else:

            return 3, None

        

        

class ReconstructMosaic(Predictor):

    """reconstruct mosaic"""



    def __init__(self, params=None, preprocess_params=None):

        super().__init__(params, preprocess_params)

        if "simple_mode" not in self.params:

            self.params["simple_mode"] = True



    def check_surface(self, image, i, j, block, color, bg, rotate):

        b = (image.shape[0] - i) // block.shape[0] + int(((image.shape[0] - i) % block.shape[0]) > 0)

        r = (image.shape[1] - j) // block.shape[1] + int(((image.shape[1] - j) % block.shape[1]) > 0)

        t = (i) // block.shape[0] + int((i) % block.shape[0] > 0)

        l = (j) // block.shape[1] + int((j) % block.shape[1] > 0)



        full_image = np.ones(((b + t) * block.shape[0], (r + l) * block.shape[1])) * color

        start_i = (block.shape[0] - i) % block.shape[0]

        start_j = (block.shape[1] - j) % block.shape[1]



        full_image[start_i : start_i + image.shape[0], start_j : start_j + image.shape[1]] = image



        for k in range(b + t):

            for n in range(r + l):

                new_block = full_image[

                    k * block.shape[0] : (k + 1) * block.shape[0], n * block.shape[1] : (n + 1) * block.shape[1]

                ]

                if (new_block == color).sum() < (block == color).sum():

                    block = new_block.copy()



        blocks = []

        for k in range(b + t):

            for n in range(r + l):

                new_block = full_image[

                    k * block.shape[0] : (k + 1) * block.shape[0], n * block.shape[1] : (n + 1) * block.shape[1]

                ]

                mask = np.logical_and(new_block != color, block != color)

                if (new_block == block)[mask].all():

                    blocks.append(new_block)

                else:

                    if rotate:

                        success = False

                        if new_block.shape[0] != new_block.shape[1]:

                            rotations = [0, 2]

                        else:

                            rotations = [0, 1, 2, 3]

                        for rotation in rotations:

                            for transpose in [True, False]:

                                rotated_block = np.rot90(new_block.copy(), rotation)

                                if transpose:

                                    rotated_block = rotated_block[::-1]

                                mask = np.logical_and(block != color, rotated_block != color)

                                if (block == rotated_block)[mask].all():

                                    blocks.append(rotated_block)

                                    success = True

                                    break

                            if success:

                                break

                        if not success:

                            return 1, None

                    else:

                        return 1, None



        new_block = block.copy()

        for curr_block in blocks:

            mask = np.logical_and(new_block != color, curr_block != color)

            if (new_block == curr_block)[mask].all():

                new_block[new_block == color] = curr_block[new_block == color]

            else:

                return 2, None



        if (new_block == color).any() and not bg:

            temp_array = np.concatenate([new_block, new_block], 0)

            temp_array = np.concatenate([temp_array, temp_array], 1)

            for k in range(new_block.shape[0]):

                for n in range(new_block.shape[1]):

                    current_array = temp_array[k : k + new_block.shape[0], n : n + new_block.shape[1]]

                    mask = np.logical_and(new_block != color, current_array != color)

                    if (new_block == current_array)[mask].all():

                        new_block[new_block == color] = current_array[new_block == color]

        if (new_block == color).any() and not bg:

            return 3, None



        for k in range(b + t):

            for n in range(r + l):

                if rotate:

                    current_array = full_image[

                        k * block.shape[0] : (k + 1) * block.shape[0], n * block.shape[1] : (n + 1) * block.shape[1]

                    ]

                    if rotate:

                        success = False

                        if current_array.shape[0] != current_array.shape[1]:

                            rotations = [0, 2]

                        else:

                            rotations = [0, 1, 2, 3]

                    for rotation in rotations:

                        for transpose in [True, False]:

                            rotated_block = np.rot90(new_block.copy(), rotation)

                            if transpose:

                                rotated_block = rotated_block[::-1]

                            mask = np.logical_and(rotated_block != color, current_array != color)

                            if (rotated_block == current_array)[mask].all():

                                full_image[

                                    k * block.shape[0] : (k + 1) * block.shape[0],

                                    n * block.shape[1] : (n + 1) * block.shape[1],

                                ] = rotated_block

                                success = True

                                break

                        if success:

                            break

                else:

                    full_image[

                        k * block.shape[0] : (k + 1) * block.shape[0], n * block.shape[1] : (n + 1) * block.shape[1]

                    ] = new_block



        result = full_image[start_i : start_i + image.shape[0], start_j : start_j + image.shape[1]]

        return 0, result



    def predict_output(self, image, params):

        """ predicts 1 output image given input image and prediction params"""

        k = 0

        itteration_list1 = list(range(2, sum(image.shape)))

        if params["big_first"]:

            itteration_list1 = list(

                range(2, (image != params["color"]).max(1).sum() + (image != params["color"]).max(0).sum() + 1)

            )

            itteration_list1 = itteration_list1[::-1]

        if params["largest_non_bg"]:

            itteration_list1 = [(image != params["color"]).max(1).sum() + (image != params["color"]).max(0).sum()]

        for size in itteration_list1:

            if params["direction"] == "all":

                itteration_list = list(range(1, size))

            elif params["direction"] == "vert":

                itteration_list = [image.shape[0]]

            else:

                itteration_list = [size - image.shape[1]]

            if params["largest_non_bg"]:

                itteration_list = [(image != params["color"]).max(1).sum()]

            for i_size in itteration_list:

                j_size = size - i_size

                if j_size < 1 or i_size < 1:

                    continue

                block = image[0 : 0 + i_size, 0 : 0 + j_size]

                status, predict = self.check_surface(

                    image, 0, 0, block, params["color"], params["have_bg"], params["rotate_block"]

                )

                if status != 0:

                    continue

                if k == params["k_th_block"]:

                    return 0, predict

                else:

                    k += 1

                    continue



        return 1, None



    def process_one_sample(self, k, initial=False):

        """ processes k train sample and updates self.solution_candidates"""

        local_candidates = []

        original_image, target_image = self.get_images(k)

        if original_image.shape != target_image.shape:

            return 1, None



        if initial:

            directions = ["all", "vert", "hor"]

            big_first_options = [True, False]

            largest_non_bg_options = [True, False]

            have_bg_options = [True, False]

            if self.params["simple_mode"]:

                rotate_block_options = [False]

                k_th_block_options = [0]

            else:

                rotate_block_options = [True, False]

                k_th_block_options = list(range(10))

        else:

            directions = list({params["direction"] for params in self.solution_candidates})

            big_first_options = list({params["big_first"] for params in self.solution_candidates})

            largest_non_bg_options = list({params["largest_non_bg"] for params in self.solution_candidates})

            have_bg_options = list({params["have_bg"] for params in self.solution_candidates})

            rotate_block_options = list({params["rotate_block"] for params in self.solution_candidates})

            k_th_block_options = list({params["k_th_block"] for params in self.solution_candidates})



        for largest_non_bg in largest_non_bg_options:

            for color in self.sample["train"][k]["colors_sorted"]:

                for direction in directions:

                    for big_first in big_first_options:

                        if largest_non_bg and not big_first:

                            continue

                        for have_bg in have_bg_options:

                            if largest_non_bg and not have_bg:

                                continue

                            if (target_image == color).any() and not have_bg:

                                continue

                            for rotate_block in rotate_block_options:

                                for k_th_block in k_th_block_options:

                                    params = {

                                        "color": color,

                                        "direction": direction,

                                        "big_first": big_first,

                                        "have_bg": have_bg,

                                        "rotate_block": rotate_block,

                                        "k_th_block": k_th_block,

                                        "largest_non_bg": largest_non_bg,

                                    }



                                    local_candidates = local_candidates + self.add_candidates_list(

                                        original_image, target_image, self.sample["train"][k], params

                                    )

        return self.update_solution_candidates(local_candidates, initial)





class ReconstructMosaicExtract(ReconstructMosaic):

    """returns the reconstructed part of the mosaic"""



    def __init__(self, params=None, preprocess_params=None):

        super().__init__(params, preprocess_params)

        if "simple_mode" not in self.params:

            self.params["simple_mode"] = True



    def predict_output(self, image, params):

        """ predicts 1 output image given input image and prediction params"""

        k = 0

        mask = image == params["color"]

        sum0 = mask.sum(0)

        sum1 = mask.sum(1)

        indices0 = np.arange(len(sum1))[sum1 > 0]

        indices1 = np.arange(len(sum0))[sum0 > 0]



        itteration_list1 = list(range(2, sum(image.shape)))

        if params["big_first"]:

            itteration_list1 = list(

                range(2, (image != params["color"]).max(1).sum() + (image != params["color"]).max(0).sum() + 1)

            )

            itteration_list1 = itteration_list1[::-1]

        for size in itteration_list1:

            if params["direction"] == "all":

                itteration_list = list(range(1, size))

            elif params["direction"] == "vert":

                itteration_list = [image.shape[0]]

            else:

                itteration_list = [size - image.shape[1]]

            for i_size in itteration_list:

                j_size = size - i_size

                if j_size < 1 or i_size < 1:

                    continue

                block = image[0 : 0 + i_size, 0 : 0 + j_size]

                status, predict = self.check_surface(

                    image, 0, 0, block, params["color"], params["have_bg"], params["rotate_block"]

                )

                if status != 0:

                    continue

                if k == params["k_th_block"]:

                    predict = predict[indices0.min() : indices0.max() + 1, indices1.min() : indices1.max() + 1]

                    return 0, predict

                else:

                    k += 1

                    continue



        return 1, None



    def process_one_sample(self, k, initial=False):

        """ processes k train sample and updates self.solution_candidates"""

        local_candidates = []

        original_image, target_image = self.get_images(k)



        if initial:

            directions = ["vert", "hor", "all"]

            big_first_options = [True, False]

            largest_non_bg_options = [True, False]

            have_bg_options = [True, False]

            if self.params["simple_mode"]:

                rotate_block_options = [False]

                k_th_block_options = [0]

            else:

                rotate_block_options = [True, False]

                k_th_block_options = list(range(10))

        else:

            directions = list({params["direction"] for params in self.solution_candidates})

            big_first_options = list({params["big_first"] for params in self.solution_candidates})

            have_bg_options = list({params["have_bg"] for params in self.solution_candidates})

            largest_non_bg_options = list({params["largest_non_bg"] for params in self.solution_candidates})

            rotate_block_options = list({params["rotate_block"] for params in self.solution_candidates})

            k_th_block_options = list({params["k_th_block"] for params in self.solution_candidates})



        for largest_non_bg in largest_non_bg_options:

            for color in self.sample["train"][k]["colors_sorted"]:

                mask = original_image == color

                sum0 = mask.sum(0)

                sum1 = mask.sum(1)



                if len(np.unique(sum0)) != 2 or len(np.unique(sum1)) != 2:

                    continue

                if target_image.shape[0] != max(sum0) or target_image.shape[1] != max(sum1):

                    continue

                for direction in directions:

                    for big_first in big_first_options:

                        if largest_non_bg and not big_first:

                            continue

                        for have_bg in have_bg_options:

                            if largest_non_bg and not have_bg:

                                continue

                            if (target_image == color).any() and not have_bg:

                                continue

                            for rotate_block in rotate_block_options:

                                for k_th_block in k_th_block_options:

                                    params = {

                                        "color": color,

                                        "direction": direction,

                                        "big_first": big_first,

                                        "have_bg": have_bg,

                                        "rotate_block": rotate_block,

                                        "k_th_block": k_th_block,

                                    }



                                    local_candidates = local_candidates + self.add_candidates_list(

                                        original_image, target_image, self.sample["train"][k], params

                                    )

        return self.update_solution_candidates(local_candidates, initial)
import json

import multiprocessing

import os

import time

import pandas as pd



import matplotlib as mpl

from matplotlib import pyplot as plt



from tqdm.notebook import tqdm

from functools import partial



import signal

import sys

import psutil





def sigterm_handler(_signo, _stack_frame):

    sys.exit(0)





def process_file(

    file_path,

    PATH,

    predictors,

    preprocess_params=None,

    color_params=None,

    show_results=True,

    break_after_answer=False,

    queue=None,

    process_whole_ds=False,

):

    with open(os.path.join(PATH, file_path), "r") as file:

        sample = json.load(file)



    submission_list = []

    sample = preprocess_sample(

        sample, params=preprocess_params, color_params=color_params, process_whole_ds=process_whole_ds

    )



    signal.signal(signal.SIGTERM, sigterm_handler)



    for predictor in predictors:

        try:

            submission_list = []

            result, answer = predictor(sample)

            if result == 0:

                if show_results:

                    show_sample(sample)



                for j in range(len(answer)):

                    answers = set([])

                    for k in range(len(answer[j])):

                        str_answer = matrix2answer(answer[j][k])

                        if str_answer not in answers:

                            if show_results and k < 3:

                                plt.matshow(answer[j][k], cmap="Set3", norm=mpl.colors.Normalize(vmin=0, vmax=9))

                                plt.show()

                                print(file_path, str_answer)

                            answers.add(str_answer)

                            submission_list.append({"output_id": file_path[:-5] + "_" + str(j), "output": str_answer})

            if queue is not None:

                queue.put(submission_list)

            if break_after_answer:

                break

        except SystemExit:

            break

    time.sleep(1)

    return





def run_parallel(

    files_list,

    PATH,

    predictors,

    preprocess_params=None,

    color_params=None,

    show_results=True,

    break_after_answer=False,

    processes=20,

    timeout=300,

    max_memory_by_process=1.4e10,

    process_whole_ds=False,

):

    process_list = []

    timing_list = []



    queue = multiprocessing.Queue(10000)

    func = partial(

        process_file,

        PATH=PATH,

        predictors=predictors,

        preprocess_params=preprocess_params,

        color_params=color_params,

        show_results=show_results,

        break_after_answer=break_after_answer,

        queue=queue,

        process_whole_ds=process_whole_ds,

    )



    result = []

    with tqdm(total=len(files_list)) as pbar:

        num_finished_previous = 0

        try:

            while True:



                num_finished = 0

                for process, start_time in zip(process_list, timing_list):

                    if process.is_alive():

                        if time.time() - start_time > timeout:

                            process.terminate()

                            while not queue.empty():

                                result = result + queue.get()

                            process.join(10)

                            print("Time out. The process is killed.")

                            num_finished += 1

                        else:

                            process_data = psutil.Process(process.pid)

                            if process_data.memory_info().rss > max_memory_by_process:

                                process.terminate()

                                while not queue.empty():

                                    result = result + queue.get()

                                process.join(10)

                                print("Memory limit is exceeded. The process is killed.")

                                num_finished += 1



                    else:

                        num_finished += 1



                if num_finished == len(files_list):

                    pbar.update(len(files_list) - num_finished_previous)

                    time.sleep(0.1)

                    break

                elif len(process_list) - num_finished < processes and len(process_list) < len(files_list):

                    p = multiprocessing.Process(target=func, args=(files_list[len(process_list)],))

                    process_list.append(p)

                    timing_list.append(time.time())

                    p.start()

                pbar.update(num_finished - num_finished_previous)

                num_finished_previous = num_finished

                # print(f"num_finished: {num_finished}, total_num: {len(process_list)}")

                while not queue.empty():

                    result = result + queue.get()

                time.sleep(0.1)

        except KeyboardInterrupt:

            for process in process_list:

                process.terminate()

                process.join(5)

            print("Got Ctrl+C")

        except Exception as error:

            for process in process_list:

                process.terminate()

                process.join(5)

            print(f"Function raised {error}")



    return result





def generate_submission(predictions_list, sample_submission_path="data/sample_submission.csv"):

    submission = pd.read_csv(sample_submission_path).to_dict("records")



    initial_ids = set([data["output_id"] for data in submission])

    new_submission = []



    ids = set([data["output_id"] for data in predictions_list])

    for output_id in ids:

        predicts = list(set([data["output"] for data in predictions_list if data["output_id"] == output_id]))

        output = " ".join(predicts[:3])

        new_submission.append({"output_id": output_id, "output": output})



    for output_id in initial_ids:

        if not output_id in ids:

            new_submission.append({"output_id": output_id, "output": ""})



    return pd.DataFrame(new_submission)





def combine_submission_files(list_of_dfs, sample_submission_path="data/sample_submission.csv"):

    submission = pd.read_csv(sample_submission_path)



    list_of_outputs = []

    for df in list_of_dfs:

        list_of_outputs.append(df.sort_values(by="output_id")["output"].astype(str).values)



    merge_output = []

    for i in range(len(list_of_outputs[0])):

        list_of_answ = [

            [x.strip() for x in output[i].strip().split(" ") if x.strip() != ""] for output in list_of_outputs

        ]

        list_of_answ = [x for x in list_of_answ if len(x) != 0]

        total_len = len(list(set([item for sublist in list_of_answ for item in sublist])))

#         print(total_len)

        while total_len > 3:

            for j in range(1, len(list_of_answ) + 1):

                if len(list_of_answ[-j]) > (j > len(list_of_answ) - 3):

                    list_of_answ[-j] = list_of_answ[-j][:-1]

                    break

            total_len = len(list(set([item for sublist in list_of_answ for item in sublist])))



        o = list(set([item for sublist in list_of_answ for item in sublist]))

        answer = " ".join(o[:3]).strip()

        while answer.find("  ") > 0:

            answer = answer.replace("  ", " ")

        merge_output.append(answer)

    submission["output"] = merge_output

    submission["output"] = submission["output"].astype(str)

    return submission
TEST_PATH = '/kaggle/input/abstraction-and-reasoning-challenge/test'

if debug:

    test_files = [x for x in os.listdir(TEST_PATH) if x[-3:]=='son'][:2]

else:

    test_files = [x for x in os.listdir(TEST_PATH) if x[-3:]=='son']

    

if fast_submit and "00576224.json" in test_files:

    test_files = test_files[:2]
params = {"skip_train": 1}

predictors= [

    ReconstructMosaic(params),

    ReconstructMosaicExtract(params),

]



preprocess_params = ["initial"]



color_params = ["coverage", "unique", "corners", "top", "grid"]



submission_list = run_parallel(

    test_files, 

    TEST_PATH, 

    predictors, 

    preprocess_params, 

    color_params, 

    timeout = 300, 

    processes = 2,

    max_memory_by_process = 0.5e+10,

    show_results = True

)



sub_df1 = generate_submission(submission_list, '/kaggle/input/abstraction-and-reasoning-challenge/sample_submission.csv')

sub_df1.head()
predictors= [Puzzle({"intersection": 0})]

                    

preprocess_params = [

    "initial",

    "block_with_side_colors",

    "min_max_blocks",

    "max_area_covered",

]



color_params = ["coverage", "unique", "corners", "top", "grid"]



submission_list = run_parallel(

    test_files, 

    TEST_PATH, 

    predictors, 

    preprocess_params, 

    color_params, 

    timeout = 300, 

    processes = 2,

    max_memory_by_process = 0.5e+10,

    show_results = True,

    process_whole_ds = True

)



sub_df2 = generate_submission(submission_list, '/kaggle/input/abstraction-and-reasoning-challenge/sample_submission.csv')

sub_df2.head()
final_submission = combine_submission_files([sub_df1,sub_df2],'/kaggle/input/abstraction-and-reasoning-challenge/sample_submission.csv')

final_submission.to_csv("submission.csv", index=False)
final_submission.head()