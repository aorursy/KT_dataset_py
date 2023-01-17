import numpy as np                              #numpy library is used to work with multidimensional array.

import pandas as pd                             #panda used for data manipulation and analysis.

                 

import os                                       #os library is used for loading file to use in the program

import json                                     #json library parses json into a string or dict, and convert string or dict to json file.

from pathlib import Path                        #support path



import matplotlib.pyplot as plt                 #support ploting a figure

from matplotlib import colors                   #colors support converting number or argument into colors



from itertools import combinations              #get different combinations of elements from numpy array.



# get the path for training_task, evaluation_task, and test_task

data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')

training_path = data_path / 'training'

evaluation_path = data_path / 'evaluation'

test_path = data_path / 'test'



#from the path above, we load the tests file's directory into our training_tasks, evaluation_tasks, and test_tasks variables

#the sorted() function is just for the list of directory to maintain some order

training_tasks = sorted(os.listdir(training_path))

evaluation_tasks = sorted(os.listdir(evaluation_path))

test_tasks = sorted(os.listdir(test_path))



#Get the first file of the training_tasks

training_task_file = str(training_path / training_tasks[0])



#Get the first file of the evaluation_tasks

evaluation_task_file = str(evaluation_path / evaluation_tasks[0])



#Get the first file of the test_tasks

test_task_file = str(test_path / test_tasks[0])



#open the file and load it

with open(training_task_file, 'r') as f:   

    #can change training_task_file to evaluation_task_file or test_task_file to have a look at evaluation file or test file

    task = json.load(f)



cmap = colors.ListedColormap(

    ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',

     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

norm = colors.Normalize(vmin=0, vmax=9)



#plotting the training task and the test task.

#use only for task in training tasks and evaluation tasks

def plot_task(task):

    n = len(task["train"]) + len(task["test"])

    fig, axs = plt.subplots(2, n, figsize=(4*n,8), dpi=50)

    plt.subplots_adjust(wspace=0, hspace=0)

    fig_num = 0

    for i, t in enumerate(task["train"]):

        t_in, t_out = np.array(t["input"]), np.array(t["output"])

        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)

        axs[0][fig_num].set_title(f'Train-{i} in')

        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))

        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))

        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)

        axs[1][fig_num].set_title(f'Train-{i} out')

        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))

        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))

        fig_num += 1

    for i, t in enumerate(task["test"]):

        t_in, t_out = np.array(t["input"]), np.array(t["output"])

        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)

        axs[0][fig_num].set_title(f'Test-{i} in')

        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))

        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))

        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)

        axs[1][fig_num].set_title(f'Test-{i} out')

        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))

        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))

        fig_num += 1

    

    plt.tight_layout()

    plt.show()

    
# plot_pictures is a function to plot our prediction of a specific task, it includes two variables pictures and labels

# pictures will be the list which contains input, output(in case of file in TEST part there is no output) and our prediction.

# labels is the list of labels "Input", "Output", "Prediction" that will be shown in the plotted figure.

def plot_pictures(pictures, labels):

        fig, axs = plt.subplots(1, len(pictures), figsize=(2 * len(pictures), 32))

        for i, (pict, label) in enumerate(zip(pictures, labels)):

            axs[i].imshow(np.array(pict), cmap=cmap, norm=norm)

            axs[i].set_title(label)

        plt.show()
file_name = "db3e9e38.json"                   #the name of the file containing the task



# Although we know that this file locates in the training set, we generalize it to make sure that no matter

# where the file locates, we can open it. How we can do it is shown in the code below.

def init_task(file_name):

    task_file = None

    task = None

    if file_name in training_tasks:

        task_file = str(training_path / file_name)

    elif file_name in evaluation_tasks:

        task_file = str(evaluation_path / file_name)

    elif file_name in test_tasks:

        task_file = str(test_path / file_name)

    with open(task_file, 'r') as f:   

        task = json.load(f)

    return task
task = init_task(file_name)

print(task)

plot_task(task)
observations = None              #Our memory, the data structure that we gonna use for remembering information

input_ = None                    #As we go through each sample in the task, we will assign the input to input_

output = None                    #As we go through each sample in the task, we will assign the output to output_

input_original = None            #As we go through each sample, we will modify the input as you can see above, so we need

                                 #a variable to keep the original sample's input, and that is input_original mission

#Every grid that we store in the memory for this task is 3x3 grid. If we enumerate the grid from top to bottom, from left to 

#right by number from 0 -> 8, the square that we care about (which is in the center) has index 4. That is what the two

#following variables mean. Stay tuned for how we will use them ...

remove_idxs = 4                  

conclusion_idx = 4



distance = 1                     #how far from a square that we want to observe and save to memory, in this case only 1 because

                                 #we only look at the squares next to it vertically, horizontally or diagonally.

k = 9                            #number of cells in the grid that we save to the memory,3x3 = 9
#_pad_image helps us to create a border of zeros around whatever numpy array that we give it.

def _pad_image(image):

    return np.pad(image, distance, constant_values=0)
#example for the usage of _pad_image

#define arr as a numpy array

arr = np.array([[1,2],[3,4]])

print(arr)

print("after using _pad_image")

print(_pad_image(arr))
def _remove_padding(frame):

    return frame[distance: -distance, distance: -distance]
def _sample_handler(sample):

    global input_, output, input_original

    input_ = np.array(sample["input"])

    input_ = _pad_image(input_)

    if "output" in sample:

        output = np.array(sample["output"])

        output = _pad_image(output)

    else:

        output = None

    input_original = input_.copy()
def _grid_walk():

    global input_

    rows, cols = input_.shape[0], input_.shape[1]

    r0 = reversed(range(distance, rows-distance))

    for i in r0:

        r1 = range(distance, cols - distance)

        for j in r1:

            yield i, j
def get_neighbours(frame, row, col):

    #get the grid (row, col) and its 8 neighbors as a 3x3 numpy array

    kernel = frame[row-distance:row+distance+1, col-distance:col+distance+1]

    #flatten the kernel and delete the value at index number remove_idxs = 4 (which is the board[i,j]'s value, not its neighbors)

    neighs = np.delete(kernel.flatten(), remove_idxs)

    return neighs
#Example for usage of get_neighbors:

#initiate a numpy array

arr = np.array([[1,2,3],[4,5,6],[7,8,9]])

print(get_neighbours(arr,1,1))
def get_label(output, row, col):

    global distance, conclusion_idx

    kernel = output[row-distance:row+distance+1, col-distance:col+distance+1]

    label = kernel.flatten()[conclusion_idx]

    return label
def _sum_neighs(neighs):

    return neighs.sum()
def _generate_observation(neighs, conclusion):

    return neighs.tolist() + [conclusion] if conclusion is not None else neighs.tolist()
def observe(task):

    global observations, input_, output, input_original

    train = task["train"].copy()                                         #get all "train" samples in a task.

    observations = []                                                    #initialize a list of observations.

    for sample in train:                                                 

        _sample_handler(sample)                                          #add padding

        for i, j in _grid_walk():                                        #walk through the grid in determined direction.

            neighs = get_neighbours(input_, i, j)                        #get neighbors' value  

            conclusion = get_label(output, i, j)                         #get the color of the examined grid from the output 

            if _sum_neighs(neighs) > 0:                                  #check if neighbors are not all black

                observation = _generate_observation(neighs, conclusion)  #get the full observation

                if observation not in observations:                      

                    observations.append(observation)                     #add the observation in memory

                input_[i, j] = output[i, j]                              #assign the output's value of (i,j) square to input_[i,j]

    input_ = input_original.copy()  # reset input
observe(task)
#LIST OF NUMBER VIEW

print(observations)
#COLOR VIEW



def unflatten(arr):

    #take the observations that have been flatten and bring it back to the 3 * 3 shape

    #initiate a 3x3 array

    three_by_three = np.zeros([3,3])

    index = 0

    for i in range(len(three_by_three)):

        for j in range(len(three_by_three[0])): 

            if i == (len(three_by_three) - 1) / 2 and i == j:

                three_by_three[i,j] = arr[-1]

            else:

                three_by_three[i, j] = arr[index]

                index += 1

    return three_by_three



def plot_array():

    global observations

    n = len(observations)

    fig, axs = plt.subplots(1, n, figsize=(4*n,8), dpi=50)

    plt.subplots_adjust(wspace=0, hspace=0)

    fig_num = 0

    for i, t in enumerate(observations):

        t_in = unflatten(observations[i])

        axs[fig_num].imshow(t_in, cmap=cmap, norm=norm)

        axs[fig_num].set_title(f'Test-{i} in')

        axs[fig_num].set_yticks(list(range(t_in.shape[0])))

        axs[fig_num].set_xticks(list(range(t_in.shape[1])))

        fig_num += 1

    

    plt.tight_layout()

    plt.show()



plot_array()

def my_prediction(task):

    global observations, input_, output, input_original

    colors = [0,7,8]

    all_predictions = []

    for sample in task["test"]:

        _sample_handler(sample)

        for i, j in _grid_walk():

            neighs = get_neighbours(input_, i, j)

            if _sum_neighs(neighs) > 0:

                for color in colors:

                    if (neighs.tolist() + [color]) in observations:

                        input_[i,j] = color

        input_ = _remove_padding(input_)

        guess = input_.tolist()

        all_predictions.append(guess)

    input_ = input_original.copy()

    return all_predictions



# And plot my predictions

def plot_predictions(task, predictions):

    n = len(task["train"]) + len(task["test"])

    fig, axs = plt.subplots(2, n, figsize=(4*n,8), dpi=50)

    plt.subplots_adjust(wspace=0, hspace=0)

    fig_num = 0

    for i, t in enumerate(task["train"]):

        t_in, t_out = np.array(t["input"]), np.array(t["output"])

        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)

        axs[0][fig_num].set_title(f'Train-{i} in')

        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))

        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))

        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)

        axs[1][fig_num].set_title(f'Train-{i} out')

        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))

        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))

        fig_num += 1

    for i, t in enumerate(task["test"]):

        t_in, t_out = np.array(t["input"]), np.array(predictions[i])

        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)

        axs[0][fig_num].set_title(f'Test-{i} in')

        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))

        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))

        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)

        axs[1][fig_num].set_title(f'My prediction-{i} out')

        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))

        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))

        fig_num += 1

    

    plt.tight_layout()

    plt.show()

    

predictions = my_prediction(task)

plot_predictions(task, predictions)     

explanations = None      #data structure we use to store our inference. Explanations will be initialized as a dictionary
def _combination_walk(k):

    r_min = 1

    r_max = 3                                   #we can even set r_max to 8 but for this task, 3 is more than enough

    indices = np.arange(k-1).tolist()           #generate a list in range(k-1), remember that k == 9

    for r in range(r_min, r_max+1):

        for combi in combinations(indices, r):  #generate all diffrent r-element sets from a list, int this case (indices)

            yield combi
def _generate_explanation(observation, combi):

    explanation = ",".join(

        [str(s) if i in combi else "-" for i, s in enumerate(observation[:k-1])])

    return explanation
obs = [0,0,0,0,0,0,0,7,8] #the 7 first elements are the features, the last element is the conclusion

combi = [7]            #hinting at position indexed 0 and 7

print(_generate_explanation(obs, combi))
# this is the function to check if our labelling we have just talked about is good or bad.

def _explanation_handler(explanations, freq_threshold):

        explanations_ = explanations.copy()          # explanations_ now store all of our labelling

        for explanation in explanations.keys():      

            if len(set(explanations[explanation]["conclusion"])) == 1:  # no contradiction condition, which means our labelling is good

                freq = len(explanations_[explanation]["conclusion"])

                if freq >= freq_threshold:

                    explanations_[explanation]["frequency"] = freq     #How many times a particular labelling happens

                                                                       #For example, in observations the explanation

                                                                       #(-,-,-,-,-,-,-,7 : 8) happens 8 times so frequency = 8

                    explanations_[explanation]["conclusion"] = int(explanations[explanation]["conclusion"][0])

                else:

                    del explanations_[explanation]

            else:

                del explanations_[explanation]

        return explanations_
def reason():

        global explanations

        freq_threshold = 2

        explanations = {}

        for combi in _combination_walk(k):

            for observation in observations:

                explanation = _generate_explanation(observation, combi)  #generate an explanation by ignore other position 

                                                                         #in observation but position in combi

                if explanation in explanations:

                    explanations[explanation]["conclusion"].append(observation[-1])  #labelling the explanation above

                                                                                     #If the explanation is already in explanations

                                                                                     #just add the label (conclusion) to it.

                else:

                    explanations[explanation] = {"conclusion": [observation[-1]]}

        explanations = _explanation_handler(explanations, freq_threshold)            #check for correctness of the explanation
reason()
print(explanations)
def _decide_conclusion(conclusions):

        conclusion = None

        val = - np.inf

        df = pd.DataFrame(conclusions, columns=["conclusion", "frequency"])

        for conc in df.conclusion.unique():

            val_ = df[(df.conclusion == conc)].frequency.shape[0]

            if val_ > val:

                conclusion, val = conc, val_

        return conclusion
# remove padding for everything

def _revert_sample_padding():

        global input_original, input_, output

        input_original = _remove_padding(input_original)

        input_ = _remove_padding(input_)

        if output is not None:

            output = _remove_padding(output)



def _compute_score(prediction):

        global output

        score = None

        if output is not None:

            _revert_sample_padding()

            score = 1 if np.array_equal(output, prediction) else 0

        return score
def plot_sample(predict=None):

    global input_original, output

    pictures = [input_original, output] if output is not None else [input_original]

    labels = ['Input', 'Output'] if output is not None else ["Input"]

    if predict is not None:

        pictures = pictures + [predict]

        labels = labels + ["Predict"]

    plot_pictures(pictures, labels)
def predict(is_train=False, visualize=False):

        global input_, explanations

        num_loops = 1

        visualize_prediction = False

        samples = task["test"] if not is_train else task["train"]

        predictions, scores = [], []

        

        for sample in samples:

            _sample_handler(sample)

            prediction = input_.copy()

            for loop in range(num_loops):

                for i, j in _grid_walk():

                    neighs = get_neighbours(prediction, i, j)

                    if _sum_neighs(neighs) > 0:

                        explanation_set, conclusions = [], []

                        for combi in _combination_walk(k):

                            observation = _generate_observation(neighs, None)          #getting the neighbours list

                            explanation = _generate_explanation(observation, combi)    #generate possible explanation by combi

                            if explanation in explanations.keys():

                                con = explanations[explanation]["conclusion"]          

                                freq = explanations[explanation]["frequency"]

                                conclusions.append((con, freq))                        #adding in possible explanations for

                                                                                       #defining the final conclusions.

                                explanation_set.append(explanation)

                        conclusion = _decide_conclusion(conclusions)                   #decide conclusion as explained above.

                        prediction[i, j] = conclusion if conclusion is not None else prediction[i, j]

                        if visualize_prediction:

                            plot_sample(prediction)

            prediction = _remove_padding(prediction)

            predictions.append(prediction)

            scores.append(_compute_score(prediction))

            if visualize:

                plot_sample(prediction)

        return predictions, scores
predict(visualize = True)