import numpy as np

import pandas as pd



import os

import json

from pathlib import Path



import matplotlib.pyplot as plt

from matplotlib import colors

import math

from pathlib import Path





data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge')

training_path = data_path / 'training'

evaluation_path = data_path / 'evaluation'

test_path = data_path / 'test'



training_tasks = sorted(os.listdir(training_path))

evaluation_tasks = sorted(os.listdir(evaluation_path))

test_tasks = sorted(os.listdir(test_path))

def plot_one(img, pred, correct):

    cmap = colors.ListedColormap(

        ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',

         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

    norm = colors.Normalize(vmin=0, vmax=9)



    fig, axs = plt.subplots(3, 1, figsize=(3 * 1, 3 * 3))

    ax = axs[0]

    input_matrix = img



    ax.imshow(input_matrix, cmap=cmap, norm=norm)

    ax.grid(True, which='both', color='lightgrey', linewidth=0.5)

    ax.set_yticks([x - 0.5 for x in range(1 + len(input_matrix))])

    ax.set_xticks([x - 0.5 for x in range(1 + len(input_matrix[0]))])

    ax.set_xticklabels([])

    ax.set_yticklabels([])

    ax.set_title("test input")



    ax = axs[1]

    # print(pred)

    # print(pred)

    input_matrix = pred

    ax.imshow(input_matrix, cmap=cmap, norm=norm)

    ax.grid(True, which='both', color='lightgrey', linewidth=0.5)

    ax.set_yticks([x - 0.5 for x in range(1 + len(input_matrix))])

    ax.set_xticks([x - 0.5 for x in range(1 + len(input_matrix[0]))])

    ax.set_xticklabels([])

    ax.set_yticklabels([])

    ax.set_title("test pred")

    # plt.show()



    ax = axs[2]

    # print(correct)

    input_matrix = correct

    ax.imshow(input_matrix, cmap=cmap, norm=norm)

    ax.grid(True, which='both', color='lightgrey', linewidth=0.5)

    ax.set_yticks([x - 0.5 for x in range(1 + len(input_matrix))])

    ax.set_xticks([x - 0.5 for x in range(1 + len(input_matrix[0]))])

    ax.set_xticklabels([])

    ax.set_yticklabels([])

    ax.set_title("test correct")

    plt.show()

def hokan_tate(img, check_color):

    """

    縦に切ります

    """

    print(check_color, "check color")

    img_old = img.copy()

    print(np.where(img == 0, 1, 0).sum())

    for j in range(4, img.shape[1]):

        data = img[:, :j]

        # print(data)

        # if j % 2 == 0:

        data1 = data[:, :j // 2]

        if j % 2 == 0:

            data2 = data[:, j // 2:][:, ::-1]

        else:

            data2 = data[:, j // 2 + 1:][:, ::-1]



        # data2 = data[:, j//2:][:, ::-1]

        not_diff = np.where(data1 == data2, 1, 0)

        # print(len(data1))

        # print(not_diff)

        wariai = not_diff.sum() / not_diff.size

        # print(wariai)

        if wariai > 0.8:

            # print(wariai)

            print(data1.shape)

            for k in range(data1.shape[0]):

                for l in range(data1.shape[1]):



                    d1 = data1[k, l]

                    d2 = data2[k, l]

                    print(d1, d2, "------------")

                    if d1 == check_color and d2 != check_color:

                        img[k, l] = d2

                    elif d2 == check_color and d1 != check_color:

                        # print(d1, d2, "-------------------")

                        # o = math.floor(l)

                        x = data1.shape[1] - l + data.shape[1] - 1

                        # print(k, x)

                        img[k, x] = d1

    # print(img)

    # plot_one(img_old, img)

    print(np.where(img == 0, 1, 0).sum())

    # return img

    # assert 0

    print("-" * 30)



    for j in range(img.shape[1], 5, -1):

        # data = img[:, h - j:]

        h = img.shape[0]

        data = img[:, h - j:]

        # print(data)

        data1 = data[:, :j // 2]

        if j % 2 == 0:

            data2 = data[:, j // 2:][:, ::-1]

        else:

            data2 = data[:, j // 2 + 1:][:, ::-1]



        # data2 = data[:, j//2:][:, ::-1]

        not_diff = np.where(data1 == data2, 1, 0)

        # print(len(data1))

        # print(not_diff)

        wariai = not_diff.sum() / not_diff.size

        print(wariai)

        if wariai > 0.8:

            # print(wariai)

            print("正解")

            print(data1.shape)

            for k in range(data1.shape[0]):

                for l in range(data1.shape[1]):

                    d1 = data1[k, l]

                    d2 = data2[k, l]

                    if d1 == check_color and d2 != check_color:

                        x = h + l - data1.shape[1] * 2

                        # print(wariai, "---------------あああああ")

                        # print(j, k, l, d1, d2, "-----------")

                        # print(data1)

                        # print(data2)

                        # print(d1)

                        # print(d2)

                        img[k, x] = d2



                    elif d2 == check_color and d1 != check_color:

                        # print(j, k, l)



                        x = h - l - 1

                        img[k, x] = d1

    # print(img)

    print(np.where(img == 0, 1, 0).sum())



    if np.where(img == 0, 1, 0).sum() == 0:

        # print(img)

        return img

    else:

        return img





def hokan_naname(img, check_color):

    """

    正方形を取ってきてx, y反転

    """

    print(check_color, "check color")

    print("斜め")

    img_old = img.copy()

    print(np.where(img == 0, 1, 0).sum())

    for j in range(4, img.shape[1]):

        data = img[img.shape[1] - j:, :j]

        # rint(data.shape)

        # print(data)

        # if j % 2 == 0:

        data1 = np.zeros_like(data)

        data2 = data.copy()

        for x in range(data.shape[0]):

            for y in range(data.shape[1]):

                data1[y, x] = data[x, y]



        # print(data1)

        # print(data2)



        # data2 = data[:, j//2:][:, ::-1]

        not_diff = np.where(data1 == data2, 1, 0)

        # print(len(data1))

        # print(not_diff)

        wariai = not_diff.sum() / not_diff.size

        # print(wariai)

        if wariai > 0.8:

            # print(wariai)

            print("正解")

            # print(data)

            # print(data1.shape)

            for k in range(data1.shape[0]):

                for l in range(data1.shape[1]):



                    d1 = data1[k, l]

                    d2 = data2[k, l]

                    # print(d1, d2)

                    # print(d1, d2, "------------")

                    if d1 == check_color and d2 != check_color:

                        data[l, k] = d2

                        img[img.shape[1] - j:, :j] = data.copy()



                    elif d2 == check_color and d1 != check_color:

                        # print(d1, d2, "-------------------")

                        # o = math.floor(l)

                        x = data1.shape[1] - l + data.shape[1] - 1

                        # print(k, x)

                        data[k, l] = d1

                        img[img.shape[1] - j:, :j] = data.copy()



    # print(img)

    # plot_one(img_old, img)

    print(np.where(img == 0, 1, 0).sum())

    # return img

    # assert 0

    print("-" * 30)



    for j in range(img.shape[1], 4, -1):

        # data = img[:, h - j:]

        h = img.shape[0]

        data = img[:j, :j]

        # print(data)

        # if j % 2 == 0:

        data1 = np.zeros_like(data)

        data2 = data.copy()

        for x in range(data.shape[0]):

            for y in range(data.shape[1]):

                data1[y, x] = data[x, y]

        # data2 = data[:, j//2:][:, ::-1]

        not_diff = np.where(data1 == data2, 1, 0)

        # print(len(data1))

        # print(not_diff)

        wariai = not_diff.sum() / not_diff.size

        print(wariai)

        if wariai > 0.8:

            # print(wariai)

            print("正解")

            print(data1.shape)

            for k in range(data1.shape[0]):

                for l in range(data1.shape[1]):

                    d1 = data1[k, l]

                    d2 = data1[l, k]

                    if d1 == check_color and d2 != check_color:

                        img[l, k] = d2



                    elif d2 == check_color and d1 != check_color:



                        img[k, l] = d1

    # print(img)

    print(np.where(img == 0, 1, 0).sum())



    if np.where(img == 0, 1, 0).sum() == 0:

        # print(img)

        return img

    else:

        return img





def hokan_yoko(img, check_color):

    """

    横に切ります

    """

    print(check_color, "check color")

    print("Yokonikiruo")

    new = np.rot90(img)

    new = hokan_tate(new, check_color)

    # print(new, "newだお")

    new = np.rot90(new, 3)

    return new
def complement(img, checkcolor):

    i = 0

    while np.where(img == checkcolor, 1, 0).sum() != 0 and i < 3:

        img = hokan_naname(img, checkcolor)

        img = hokan_tate(img, checkcolor)

        img = hokan_yoko(img, checkcolor)

        i += 1

    return img
class Features:

    def __init__(self, task):

        self.task = task

        self.df = pd.DataFrame()



    def features(self):

        # print(self.task["train"])

        for j, t in enumerate(self.task["train"]):

            t_in, t_out = np.array(t["input"]), np.array(t["output"])



            """

            命名規則はノリが多いが、inputとoutputで同じ特徴量を作成するときは頭にin outをつける

            

            A_eq_BはAとBが等しいかどうか

            

            など算術記号はノリが多いので感覚で読んでもらいたいがノリで読めないものにはコメントを付けている

            """



            """

            サイズに関する特徴量

            """



            self.df.loc[j, "in_height"] = t_in.shape[0]

            self.df.loc[j, "in_width"] = t_in.shape[1]

            self.df.loc[j, "out_height"] = t_out.shape[0]

            self.df.loc[j, "out_width"] = t_out.shape[1]



            self.df.loc[j, "in_height_eq_width"] = t_in.shape[0] == t_in.shape[1]

            self.df.loc[j, "out_height_eq_width"] = t_out.shape[0] == t_out.shape[1]



            self.df.loc[j, "in_out_height_eq"] = t_in.shape[0] == t_out.shape[0]

            self.df.loc[j, "in_out_width_eq"] = t_in.shape[1] == t_out.shape[1]



            self.df.loc[j, "height_out_div_in_R"] = t_out.shape[0] % t_in.shape[0]

            self.df.loc[j, "width_out_div_in_R"] = t_out.shape[1] % t_in.shape[1]



            """

            色に関する特徴量

            """

            in_uq = np.unique(t_in)

            out_uq = np.unique(t_out)

            self.df.loc[j, "in_unique_colors"] = len(in_uq)

            self.df.loc[j, "out_unique_colors"] = len(out_uq)



            self.df.loc[j, "in_unique_colors_off_black"] = len(in_uq[in_uq != 0])

            self.df.loc[j, "out_unique_colors_off_black"] = len(out_uq[out_uq != 0])



            # self.df.loc[j, "in_unique_colors"] = in_uq

            # self.df.loc[j, "out_unique_colors"] = out_uq

            for i in range(10):

                name = "colors_diff_" + str(i)

                self.df.loc[j, name] = not (i in in_uq) ^ (i in out_uq)



            """

            ブロックに関する特徴量

            """

            in_blocks = np.where(t_in == 0, 0, 1)

            out_blocks = np.where(t_out == 0, 0, 1)

            self.df.loc[j, "in_blocks_num"] = np.sum(in_blocks)

            self.df.loc[j, "out_blocks_num"] = np.sum(out_blocks)

            # ブロック数と全体のマスの比率

            self.df.loc[j, "in_blocks_ratio"] = np.sum(in_blocks) / len(in_blocks)

            self.df.loc[j, "out_blocks_ratio"] = np.sum(out_blocks) / len(out_blocks)



            self.df.loc[j, "ratio_in_eq_out"] = (np.sum(in_blocks) / len(in_blocks)) == (

                    np.sum(out_blocks) / len(out_blocks))



        all_true_columns = np.zeros(len(self.df.columns))



        for i, x in enumerate(self.df.columns):

            if self.df[x].all():

                all_true_columns[i] = 1



        self.df.loc[len(self.task["train"]), :] = all_true_columns

        # print(all_true_columns)

        # print(self.df)



        return self.df



    def tentaisho(self, img):

        o_x, o_y = img.shape[0] / 2, img.shape[1] / 2

        new = np.zeros_like(img)

        for i in range(img.shape[0]):

            for j in range(img.shape[1]):

                x = o_x + o_x - i

                y = o_y + o_y - i

                new[i, j] = img[x, y]

        return new





def rule(data_old, test_task):

    test_task = np.array(test_task)

    data = data_old.iloc[:-1]

    bool_data = data_old.iloc[-1]

    if np.all(bool_data.iloc[:6] == 1):

        if data.loc[0, "in_height"] > 15:

            print("補完ですね")

            check_color = 0

            for j in range(10):

                name = "colors_diff_" + str(j)

                # print(name)



                if bool_data[name] == 0:

                    check_color += j

                    break

            # plot_one(test_task, test_task, test_task)

            img = complement(test_task, check_color)

            # return img

            return img

        return test_task

    return test_task

def main(i):

    path = training_tasks[i]

    task_file = str(training_path / path)



    with open(task_file, "r") as f:

        task = json.load(f)



    fea = Features(task)

    test_task = task["test"][0]["input"]

    bool_fe = fea.features()

    pred = rule(bool_fe, test_task)

    test_y = np.array(task["test"][0]["output"])

    print((test_y == pred).all())

    print("DONE")

    

    return test_task, pred, test_y
test_task, pred, test_y = main(16)
plot_one(test_task, pred, test_y)
test_task, pred, test_y = main(60)
plot_one(test_task, pred, test_y)
test_task, pred, test_y = main(174)
plot_one(test_task, pred, test_y)
test_task, pred, test_y = main(73)
plot_one(test_task, pred, test_y)