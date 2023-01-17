import math

import matplotlib.pyplot as plt

import pandas as pd

from collections import defaultdict

import sklearn.linear_model as lm

import sklearn.metrics as met

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.utils import shuffle

from typing import Tuple



%matplotlib inline
# Load in the pulsar star data

pulsar_stars = pd.read_csv("../input/predicting-a-pulsar-star/pulsar_stars.csv")



# Rename the columns so that they can be displayed in a collection of subplots

dataset_columns = ["$\mu$ Int",

                   "$\sigma$ Int",

                   "Kurt Int",

                   "Skew Int",

                   "$\mu$ DM-SNR",

                   "$\sigma$ DM-SNR",

                   "Kurt DM-SNR",

                   "Skew DM-SNR",

                   "Target class"]



# Define the columns corresponding to the features and target

feature_columns = dataset_columns[:-1]

value_label = dataset_columns[-1]

value_description = "Target value"

value_dict = {0: "Non-pulsar", 1: "Pulsar"}



pulsar_stars.columns = dataset_columns
# Normalise the values in the dataset

normalised_dataset = pulsar_stars.loc[:, feature_columns]

normalised_dataset = StandardScaler().fit_transform(normalised_dataset)



# Store in a data frame and add in the target values

normalised_df = pd.DataFrame(normalised_dataset, columns=feature_columns)

normalised_df[value_label] = pulsar_stars[value_label]
# Plot the relationship between each pair of features using a collection of subplots

alpha = 0.05

colors = ["r", "b"]



num_categories = len(feature_columns)

num_permutations = int(num_categories * (num_categories - 1) / 2) # If there are n categories, this is the (n-1)th triangular number

num_cols = 4

num_rows = int(math.ceil(num_permutations / num_cols))            # Display four plots on each row of the figure



fig, ax = plt.subplots(num_rows, num_cols, figsize=(11.69, 8.27 * num_rows / 2)) # Defined so that two rows of plots would take up a single landscape page of A4 paper



row = 0

column = 0



for feature1_index in range(num_categories - 1):

    feature1 = feature_columns[feature1_index]

    for feature2_index in range(feature1_index + 1, num_categories):

        feature2 = feature_columns[feature2_index]

        

        color_index = 0

        max_color_index = len(colors) - 1

        

        for target_value, description in value_dict.items():

            sub_df = normalised_df.loc[normalised_df[value_label] == target_value]

            

            ax[row, column].scatter(sub_df[feature1], sub_df[feature2], c=colors[color_index], label=description, alpha=alpha)

            ax[row, column].set_title(f"{feature2}/{feature1}")

            

            color_index += 1

        if column < num_cols - 1:

            column += 1

        else:

            row += 1

            column = 0

legend = ax[-1, -1].legend(loc="best")

for lh in legend.legendHandles:

    lh.set_alpha(1)



plt.show()

plt.close()
dataset_size = len(normalised_df)

seed = 0



shuffled_set = shuffle(normalised_df, random_state=seed)



training_set_fraction = 0.7

training_set_size = int(dataset_size * training_set_fraction)



test_set_size = int(math.ceil((1 - training_set_fraction) * dataset_size / 2))

test_set_split = training_set_size + test_set_size



training_set_df = shuffled_set.iloc[:training_set_size, :]

test_set_df = shuffled_set.iloc[training_set_size:test_set_split, :]

validation_set_df = shuffled_set.iloc[test_set_split:, :]



assert len(training_set_df) + len(test_set_df) + len(validation_set_df) == dataset_size 
def fit_and_assess_data(training_x_data: pd.DataFrame,

                        training_y_data: pd.DataFrame,

                        test_x_data: pd.DataFrame,

                        test_y_data: pd.DataFrame,

                        k: int) -> Tuple[float, float, float, float]:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(training_x_data, training_y_data)

    

    predictions = knn.predict(test_x_data)

    

    confusion_matrix = defaultdict(int)

    for i in range(len(test_y_data)):

        fitted_value = value_dict[predictions[i]]

        actual_value = value_dict[test_y_data.iloc[i]]

        confusion_matrix[(fitted_value, actual_value)] += 1



    num_correct_pulsars = confusion_matrix[("Pulsar", "Pulsar")]

    precision = num_correct_pulsars / (num_correct_pulsars + confusion_matrix[("Pulsar", "Non-pulsar")])

    recall = num_correct_pulsars / (num_correct_pulsars + confusion_matrix["Non-pulsar", "Pulsar"])

    f1_score = 2 * precision * recall / (precision + recall)

    

    return knn.score(test_x_data, test_y_data), precision, recall, f1_score
min_k = 1

max_k = 100



best_k = None

best_f1_score = 0

k_accuracy = 0

k_precision = 0

k_recall = 0



k_values = []

accuracies = []

precisions = []

recalls = []

f1_scores = []



training_x_data = training_set_df.loc[:, feature_columns]

training_y_data = training_set_df.loc[:, value_label]



test_x_data = test_set_df.loc[:, feature_columns]

test_y_data = test_set_df.loc[:, value_label]



print(f"Running k-nearest neighbours for k = {min_k}, ..., {max_k}.")

for k in range(min_k, max_k + 1):

    accuracy, precision, recall, f1_score = fit_and_assess_data(training_x_data, training_y_data, test_x_data, test_y_data, k)

    k_values.append(k)

    accuracies.append(accuracy)

    precisions.append(precision)

    recalls.append(recall)

    f1_scores.append(f1_score)

    

    if f1_score > best_f1_score:

        best_k = k

        k_accuracy = accuracy

        k_precision = precision

        k_recall = recall

        best_f1_score = f1_score



print(f"The chosen value of k is {best_k}:")

print(f"Accuracy = {k_accuracy:.2%}.")

print(f"Precision = {k_precision:.2%}.")

print(f"Recall = {k_recall:.2%}.")

print(f"F1 score = {best_f1_score:.2%}.")



plt.figure(figsize=(11.69, 8.27))

plt.plot(k_values, accuracies, "r-", label="Accuracy")

plt.plot(k_values, precisions, "g-", label="Precision")

plt.plot(k_values, recalls, "b-", label="Recall")

plt.plot(k_values, f1_scores, "k-", label="$F1$ score")

plt.xlabel("$k$ value")

plt.ylabel("Fractional value between 0 and 1")

plt.grid(which="major", axis="both")

plt.legend(loc="best")

plt.title("Plot of kNN accuracy for increasing values of $k$")



plt.show()

plt.close()
validation_x_data = validation_set_df.loc[:, feature_columns]

validation_y_data = validation_set_df.loc[:, value_label]



accuracy, precision, recall, f1_score = fit_and_assess_data(training_x_data, training_y_data, validation_x_data, validation_y_data, best_k)



print(f"Running k-nearest neighbours against the validation data for k = {best_k}:")

print(f"Accuracy = {accuracy:.2%}.")

print(f"Precision = {precision:.2%}.")

print(f"Recall = {recall:.2%}.")

print(f"F1 score = {f1_score:.2%}.")
alpha_steps = 200

max_alpha = 2.0

min_alpha = max_alpha / alpha_steps



best_alpha = None

best_f1_score = 0

alpha_accuracy = 0

alpha_precision = 0

alpha_recall = 0



alphas = [i * min_alpha for i in range(1, alpha_steps + 1)]

accuracies = []

precisions = []

recalls = []

f1_scores = []



print(f"Running regularised logistic regression for alpha = {min_alpha}, ..., {max_alpha}.")

for alpha in alphas:

    clf = lm.LogisticRegression(random_state=seed, solver="sag", C=1 / alpha)

    clf.fit(training_x_data, training_y_data)

    

    test_y_data_fitted = clf.predict(test_x_data)

    tn, fp, fn, tp = met.confusion_matrix(test_y_data, test_y_data_fitted).ravel()

    

    accuracy = (tp + tn) / (tn + fp + fn + tp)

    precision = tp / (tp + fp)

    recall = tp / (tp + fn)

    f1_score = 2 * precision * recall / (precision + recall)

    

    accuracies.append(accuracy)

    precisions.append(precision)

    recalls.append(recall)

    f1_scores.append(f1_score)

    

    if f1_score > best_f1_score:

        best_alpha = alpha

        alpha_accuracy = accuracy

        alpha_precision = precision

        alpha_recall = recall

        best_f1_score = f1_score



print(f"The chosen value of alpha is {best_alpha:.2f}:")

print(f"Accuracy = {alpha_accuracy:.2%}.")

print(f"Precision = {alpha_precision:.2%}.")

print(f"Recall = {alpha_recall:.2%}.")

print(f"F1 score = {best_f1_score:.2%}.")



plt.figure(figsize=(11.69, 8.27))

plt.plot(alphas, accuracies, "r-", label="Accuracy")

plt.plot(alphas, precisions, "g-", label="Precision")

plt.plot(alphas, recalls, "b-", label="Recall")

plt.plot(alphas, f1_scores, "k-", label="$F1$ score")

plt.xlabel("$\\alpha$ value")

plt.ylabel("Fractional value between 0 and 1")

plt.grid(which="major", axis="both")

plt.legend(loc="best")

plt.title("Plot of logistic regression accuracy for increasing values of $\\alpha$")



plt.show()

plt.close()
clf = lm.LogisticRegression(random_state=seed, solver="sag", C=best_alpha)

clf.fit(training_x_data, training_y_data)



validation_y_data_fitted = clf.predict(validation_x_data)

tn, fp, fn, tp = met.confusion_matrix(validation_y_data, validation_y_data_fitted).ravel()



accuracy = (tp + tn) / (tn + fp + fn + tp)

precision = tp / (tp + fp)

recall = tp / (tp + fn)

f1_score = 2 * precision * recall / (precision + recall)



print(f"Running regularised logistic regression against the validation data for alpha = {best_alpha:.2f}:")

print(f"Accuracy = {accuracy:.2%}.")

print(f"Precision = {precision:.2%}.")

print(f"Recall = {recall:.2%}.")

print(f"F1 score = {f1_score:.2%}.")