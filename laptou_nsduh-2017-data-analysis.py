import numpy as np

import pandas as pd



%matplotlib inline

import matplotlib.pyplot as plt

from scipy import stats



# use seaborn plotting defaults

import seaborn as sns; sns.set()



raw_data = pd.read_csv("../input/NSDUH_2017_Tab.tsv", sep="\t", dtype=np.str)
cols = [

    "AGE2", # age

    "IRSEX", # gender

    "NEWRACE2", # race

    "SEXIDENT", # sexual identity

    "EDUSCHGRD2", # grade level

    "MILTFAMLY", # family in military?

    "YESCHFLT", # how did you feel about going to school?

    "YESCHWRK", # how often is your schoolwork meaningful?

    "YESCHIMP", # how important are the things you have yet to learn?

    "YETCGJOB", # how often does your teacher praise you?

    "YESTSCIG", # how many of your peers smoke cigarettes?

    "YESTSMJ", # how many of your peers smoke marijuana?

    "YESTSALC", # how many of your peers drink alcohol?

    "YESTSDNK", # how many of your peers get drunk weekly?

    "YEPCHORE", # do your parents make you do chores?

    "YEPGDJOB", # how often do your parents praise you?

    "YEPPROUD", # how often do your parents tell you they are proud of you?

    "YEYARGUP", # how often do you fight with your parents?

    "YEYFGTSW", # how often do you get into serious fights at school?

    "YETLKNON", # there is no one you can talk to about serious problems

    "YETLKPAR", # there a parent you can talk to about serious problems

    "YETLKBGF", # there an s/o you can talk to about serious problems

    "YETLKOTA", # there another adult you can talk to about serious problems

    "YETLKSOP", # there another person you can talk to about serious problems

    "YESCHACT", # how many school extracurriculars?

    "YECOMACT", # how many community extracurriculars?

    "YEFAIACT", # how many church extracurriculars?

    "DSTNRV30", # how often have you felt nervous in the past 30 days?

    "DSTHOP30", # how often have you felt hopeless in the past 30 days?

    "DSTCHR30", # how often have you felt cheerless in the past 30 days?

    "DSTEFF30", # how often have you felt listless in the past 30 days?

    "DSTNGD30", # how often have you felt worthless in the past 30 days?

    "IRFAMSOC", # family on welfare?

    "IRFAMIN3", # family income tier

    "CIGREC", # how long has it been since you had a cigarette?

    "SMKLSSREC", # how long has it been since you had smokeless tobacco (snuff, etc)?

    "CIGARREC", # how long has it been since you had a cigar?

    "ALCREC", # how long has it been since you have had alcohol?

    "MJREC", # how long has it been since you have had marijuana?

    "OXCNNMYR", # used oxycontin in a non-prescription way?

    "PNRNMREC", # how long has it been since you misused a pain reliever?

]



raw_data_picked = raw_data[cols].apply(pd.to_numeric)
raw_data_picked = raw_data_picked.loc[(raw_data_picked["EDUSCHGRD2"] >= 5) &  (raw_data_picked["EDUSCHGRD2"] <= 8)]
adjusted_data = pd.DataFrame({ 

    "age": raw_data_picked["AGE2"] + 11,

    "sex": raw_data_picked["IRSEX"] - 1,

    "race": raw_data_picked["NEWRACE2"],

    "sex_orientation": raw_data_picked["SEXIDENT"],

    "grade_level": raw_data_picked["EDUSCHGRD2"] + 4,

    "military_family": raw_data_picked["MILTFAMLY"] == 1,

    "school_feeling": raw_data_picked["YESCHFLT"],

    "school_work_meaningful": raw_data_picked["YESCHWRK"],

    "school_important": raw_data_picked["YESCHIMP"],

    "teacher_praise": raw_data_picked["YETCGJOB"],

    "peers_cigarettes": raw_data_picked["YESTSCIG"],

    "peers_marijuana": raw_data_picked["YESTSMJ"],

    "peers_alcohol": raw_data_picked["YESTSALC"],

    "peers_alcohol_weekly": raw_data_picked["YESTSDNK"],

    "chores": raw_data_picked["YEPCHORE"],

    "parent_praise": raw_data_picked["YEPGDJOB"],

    "parent_proud": raw_data_picked["YEPPROUD"],

    "parent_fight": raw_data_picked["YEYARGUP"],

    "fight": raw_data_picked["YEYFGTSW"],

    "felt_nervous": raw_data_picked["DSTNRV30"],

    "felt_hopeless": raw_data_picked["DSTHOP30"],

    "felt_cheerless": raw_data_picked["DSTCHR30"],

    "felt_listless": raw_data_picked["DSTEFF30"],

    "felt_worthless": raw_data_picked["DSTNGD30"],

    "school_extracurriculars": raw_data_picked["YESCHACT"],

    "community_extracurriculars": raw_data_picked["YECOMACT"],

    "church_extracurriculars": raw_data_picked["YEFAIACT"],

    "welfare": raw_data_picked["IRFAMSOC"] == 1,

    "income_tier": raw_data_picked["IRFAMIN3"],

    "last_cigarette": raw_data_picked["CIGREC"],

    "last_smokeless_tobacco": raw_data_picked["SMKLSSREC"],

    "last_cigar": raw_data_picked["CIGARREC"],

    "last_nicotine": raw_data_picked[["CIGREC", "SMKLSSREC", "CIGARREC"]].min(axis=1),

    "last_alcohol": raw_data_picked["ALCREC"],

    "last_marijuana": raw_data_picked["MJREC"],

    "last_opioid": raw_data_picked["PNRNMREC"],

})



adjusted_data
from sklearn.preprocessing import label_binarize



n_samples = adjusted_data.shape[0]



age_bin = label_binarize(adjusted_data["age"], range(13, 19))

sex_orientation_bin = label_binarize(adjusted_data["sex_orientation"], range(1, 4))

grade_level_bin = label_binarize(adjusted_data["grade_level"], range(9, 13))

race_bin = label_binarize(adjusted_data["race"], range(1, 8))



input_vec = np.hstack((

    age_bin, 

    adjusted_data["sex"].values[:, np.newaxis] - 1, # we need the same number of dimensions in every array for hstack

    sex_orientation_bin,

    grade_level_bin,

    race_bin,

    adjusted_data["military_family"].values[:, np.newaxis],

    adjusted_data["school_feeling"].values[:, np.newaxis],

    adjusted_data["school_work_meaningful"].values[:, np.newaxis],

    adjusted_data["school_important"].values[:, np.newaxis],

    adjusted_data["teacher_praise"].values[:, np.newaxis],

    adjusted_data["peers_cigarettes"].values[:, np.newaxis],

    adjusted_data["peers_marijuana"].values[:, np.newaxis],

    adjusted_data["peers_alcohol"].values[:, np.newaxis],

    adjusted_data["peers_alcohol_weekly"].values[:, np.newaxis],

    adjusted_data["felt_nervous"].values[:, np.newaxis],

    adjusted_data["felt_hopeless"].values[:, np.newaxis],

    adjusted_data["felt_cheerless"].values[:, np.newaxis],

    adjusted_data["felt_listless"].values[:, np.newaxis],

    adjusted_data["felt_worthless"].values[:, np.newaxis],

    adjusted_data["chores"].values[:, np.newaxis],

    adjusted_data["parent_praise"].values[:, np.newaxis],

    adjusted_data["parent_proud"].values[:, np.newaxis],

    adjusted_data["parent_fight"].values[:, np.newaxis],

    adjusted_data["fight"].values[:, np.newaxis],

    adjusted_data["school_extracurriculars"].values[:, np.newaxis],

    adjusted_data["community_extracurriculars"].values[:, np.newaxis],

    adjusted_data["church_extracurriculars"].values[:, np.newaxis],

    adjusted_data["welfare"].values[:, np.newaxis],

    adjusted_data["income_tier"].values[:, np.newaxis] / 7.0))

print(input_vec.shape)



# value 91 = never used drug

output_vec = adjusted_data.loc[:, "last_cigarette":"last_opioid"] != 91
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split



# alcohol input data should not include whether the student 



X_train, X_test, y_train, y_test = train_test_split(input_vec, output_vec["last_alcohol"].values)

X_train.shape
alcohol_svm = SVC(kernel = "rbf", C = 1e4, gamma = 1)

alcohol_svm.fit(X_train, y_train)
from sklearn.metrics import confusion_matrix



y_pred = alcohol_svm.predict(X_test)



cm = confusion_matrix(y_test, y_pred, labels = [0, 1])

sns.heatmap(cm, annot = True, fmt = "d")
from sklearn.neural_network import MLPClassifier



alcohol_nn = MLPClassifier(alpha=1e-6, hidden_layer_sizes=(120, 30), random_state = 1, activation = "relu", max_iter = 600)

alcohol_nn.fit(X_train, y_train)



y_re = alcohol_nn.predict(X_train)

y_pred = alcohol_nn.predict(X_test)



print(f"success rate: {np.sum(y_pred == y_test) / y_test.shape[0] * 100:2f}%")

sns.heatmap(confusion_matrix(y_test, y_pred, labels = [0, 1]), annot = True, fmt = "d")
print(f"self-accuracy: {np.sum(y_train == y_re) / y_re.shape[0] * 100:2f}%")

sns.heatmap(confusion_matrix(y_train, y_re, labels = [0, 1]), annot = True, fmt = "d")
X_train, X_test, y_train, y_test = train_test_split(input_vec, output_vec[["last_alcohol", "last_nicotine", "last_opioid", "last_marijuana"]].values != 1)



import torch

import torch.autograd as tgd

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim



X_train_tensor = torch.tensor(X_train, requires_grad = True, dtype = torch.float)

X_test_tensor = torch.tensor(X_test, requires_grad = True, dtype = torch.float)

y_train_tensor = torch.tensor(y_train.astype(np.float32))

y_test_tensor = torch.tensor(y_test.astype(np.float32))

y_test_tensor.shape
# we can use convenience function because this network does not require

# convolution or other advanced features

alcohol_nn = nn.Sequential(

    nn.Linear(input_vec.shape[1], 120),

    nn.Sigmoid(),

    nn.Dropout(p = 0.3),

    nn.Linear(120, 4),

    # nn.ReLU(),

    # nn.Linear(60, 1),

    nn.Sigmoid()) 
criterion = nn.BCELoss()

optimiser = optim.Adam(alcohol_nn.parameters(), lr = 0.005)



alcohol_nn.train()

for i in range(600):

    output = alcohol_nn(X_train_tensor) # calculate our predicted values 

    loss = criterion(output, y_train_tensor) # measure how far off they are

    

    optimiser.zero_grad() # zero the gradient buffers

    loss.backward() # backpropagate the error

    optimiser.step() # adjust the weights and biases

    

    if loss < 1e-6:

        break

        

y_re = alcohol_nn(X_train_tensor).detach().numpy().ravel() > 0.5

y_pred = alcohol_nn(X_test_tensor).detach().numpy().ravel() > 0.5
print(f"loss: {loss}")

print(f"success rate: {np.sum(y_pred == y_test.ravel()) / y_test.size * 100:2f}%")

sns.heatmap(confusion_matrix(y_test.ravel(), y_pred.ravel(), labels = [0, 1]), annot = True, fmt = "d")
print(f"self-accuracy: {np.sum(y_re.ravel() == y_train) / y_train.shape[0] * 100:2f}%")

sns.heatmap(confusion_matrix(y_train.ravel(), y_re, labels = [0, 1]), annot = True, fmt = "d")
list(adjusted_data)