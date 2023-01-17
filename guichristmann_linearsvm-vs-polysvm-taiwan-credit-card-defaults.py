import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.svm import SVC, LinearSVC

from sklearn.metrics import classification_report

from sklearn.cluster import KMeans

sns.set_style("whitegrid")
dataset = pd.read_csv("../input/UCI_Credit_Card.csv")

print(dataset.head())

print(f"Amount of samples: {dataset.shape[0]}")
def plotDist(dataset):

    features_to_plot = ["SEX", "EDUCATION", "MARRIAGE", "default.payment.next.month"]

    # Define integer to string mappings for a pretty graph

    intToStr = {"SEX": {1: "Male", 2: "Female"},

                "EDUCATION": {1: "Graduate School", 2: "University", 3: "High School", 4: "Other", 5: "Unknown", 6: "Unknown", 0: "Unknown"},

                "MARRIAGE": {1: "Married", 2: "Single", 3: "Other", 0: "Unknown"},

                "default.payment.next.month": {0: "No", 1: "Yes"}

               }

    

    # Iterate the specified features

    for f in features_to_plot:

        count = {} # Use dictionary to count

        for i, s in enumerate(dataset[f]):

            # Manually replace the number with the nominal string for a pretty graph

            if f in intToStr.keys():

                s = intToStr[f][s]



            if s in count.keys():

                count[s] += 1

            else:

                count[s] = 1



        values = np.array(list(count.values()))

        keys = list(count.keys())

        

        # Plot graph

        fig, ax = plt.subplots(figsize=(13, 4))

        sns.barplot(keys, values)

        plt.title(f)

        plt.ylabel("Number of samples")

        plt.show()

        

plotDist(dataset)
# Normalize columns 12 to 23. BILL_AMT1 to BILL_AMT6 and PAY_AMT1 to PAY_AMT6

norm_columns = ["LIMIT_BAL", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",

                "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "AGE"]

for c in norm_columns:

    max_val = np.max(dataset[c])

    min_val = np.min(dataset[c])

    

    dataset[c] = (dataset[c] - min_val) * 2 / (max_val - min_val) - 1



dataset.drop(columns=["ID"], inplace=True)

print(dataset.head())
def trainTestSplit(dataset, valid_per=0.1):

    n_samples = dataset.shape[0] # Total number of samples

    n_val = int(valid_per * n_samples)

    

    indices = np.arange(0, n_samples) # Generate a big array with all indices

    np.random.shuffle(indices) # Shuffle the array, numpy shuffles inplace

    

    # Perform the splits

    x_train = dataset.iloc[indices[n_val:], :-1].values # Last column is the feature we want to predict

    y_train = dataset.iloc[indices[n_val:], -1].values

    x_test = dataset.iloc[indices[:n_val], :-1].values

    y_test = dataset.iloc[indices[:n_val], -1].values

    

    return x_train, y_train, x_test, y_test
print("Getting new dataset split...")

# Get a dataset split for training and validation

x_train, y_train, x_test, y_test = trainTestSplit(dataset, 0.2)



print(f"Training on {x_train.shape[0]} samples...")

print("\n#### Linear SVM Results ####")

# Create Linear SVM model

lsvm = LinearSVC(max_iter=32000) # If we don't specify anything it assumed all classes have same weight

lsvm.fit(x_train, y_train)

y_pred = lsvm.predict(x_test)

linear_acc = lsvm.score(x_test, y_test)

print(f"Linear SVM Acc: {linear_acc*100} % - Validated on {y_test.shape[0]} samples")

print(classification_report(y_test, y_pred))



print("\n#### Polynomial SVM with Degree 3 Results ####")

# Create Polynomial SVM

svm = SVC(gamma='scale', kernel='poly', degree=3)

svm.fit(x_train, y_train)

poly_acc = svm.score(x_test, y_test)

y_pred = svm.predict(x_test)

print(f"Polynomial SVM Acc: {poly_acc*100} %")

print(classification_report(y_test, y_pred))
print(f"Training on {x_train.shape[0]} samples...")

print("\n#### Linear SVM Results ####")

# Create Linear SVM model

lsvm = LinearSVC(max_iter=32000, class_weight="balanced") # Compute weight based on sample count per class

lsvm.fit(x_train, y_train)

y_pred = lsvm.predict(x_test)

linear_acc = lsvm.score(x_test, y_test)

print(f"Linear SVM Acc: {linear_acc*100} % - Validated on {y_test.shape[0]} samples")

print(classification_report(y_test, y_pred))



print("\n#### Polynomial SVM with Degree 3 Results ####")

# Create Polynomial SVM

svm = SVC(gamma='scale', kernel='poly', degree=3, 

          class_weight="balanced") # Compute weight based on sample count per class

svm.fit(x_train, y_train)

poly_acc = svm.score(x_test, y_test)

y_pred = svm.predict(x_test)

print(f"Polynomial SVM Acc: {poly_acc*100} %")

print(classification_report(y_test, y_pred))
def balanceTrainSet(x_train, y_train):

    samples_per_class = np.bincount(y_train) # Count samples per class

    dom_class = np.argmax(samples_per_class) # Max class index

    min_class = np.argmin(samples_per_class) # Min class index

    n_min = samples_per_class[min_class] # Number of samples in min class

    

    # Get indices for the dominant and the minor class

    dom_indices = np.where(y_train == dom_class)[0]

    min_indices = np.where(y_train == min_class)[0]

    np.random.shuffle(dom_indices) # Shuffle dom_indices

    # Contatenate both indices, using only the same number of indices from dom_indices as in min_indices

    indices = np.concatenate([min_indices, dom_indices[:n_min]], axis=0)

    np.random.shuffle(indices)

    

    # Build the new training set

    new_x_train = x_train[indices]

    new_y_train = y_train[indices]

    

    return new_x_train, new_y_train

    

bal_x_train, bal_y_train = balanceTrainSet(x_train, y_train)



print(f"Training on {bal_x_train.shape[0]} samples...")

print("\n#### Linear SVM Results ####")

# Create Linear SVM model

lsvm = LinearSVC(max_iter=32000) # If we don't specify anything it assumed all classes have same weight

lsvm.fit(bal_x_train, bal_y_train)

y_pred = lsvm.predict(x_test)

linear_acc = lsvm.score(x_test, y_test)

print(f"Linear SVM Acc: {linear_acc*100} % - Validated on {y_test.shape[0]} samples")

print(classification_report(y_test, y_pred))



print("\n#### Polynomial SVM with Degree 3 Results ####")

# Create Polynomial SVM

svm = SVC(gamma='scale', kernel='poly', degree=3)

svm.fit(bal_x_train, bal_y_train)

poly_acc = svm.score(x_test, y_test)

y_pred = svm.predict(x_test)

print(f"Polynomial SVM Acc: {poly_acc*100} %")

print(classification_report(y_test, y_pred))
N_ROUNDS = 10

l_reports = [] # Linear SVM reports

p_reports = [] # Polynomial SVM reports

l_accs = [] # Linear SVM accuracy history

p_accs = [] # Polynomial SVM accuracy history

for i in range(N_ROUNDS):

    print(f"### Round {i+1} ###")

    print("Getting new dataset split...")

    x_train, y_train, x_test, y_test = trainTestSplit(dataset, 0.2)    

    print(f"Training on {x_train.shape[0]} samples...")

    

    # Create a new Linear SVM model

    lsvm = LinearSVC(max_iter=32000, class_weight="balanced") # Compute weight based on sample count per class

    lsvm.fit(x_train, y_train)

    y_pred = lsvm.predict(x_test)

    linear_acc = lsvm.score(x_test, y_test)

    print(f"Linear SVM Acc: {linear_acc*100} % - Validated on {y_test.shape[0]} samples")

    report = classification_report(y_test, y_pred, output_dict=True)

    l_reports.append(report)

    l_accs.append(linear_acc)



    # Create Polynomial SVM

    svm = SVC(gamma='scale', kernel='poly', degree=3, 

              class_weight="balanced") # Compute weight based on sample count per class

    svm.fit(x_train, y_train)

    poly_acc = svm.score(x_test, y_test)

    y_pred = svm.predict(x_test)

    print(f"Polynomial SVM Acc: {poly_acc*100} %")

    report = classification_report(y_test, y_pred, output_dict=True)

    p_reports.append(report)

    p_accs.append(poly_acc)

    

print("### Finished ###")

print("Linear SVM Results:")

print(l_reports[0])

mean_acc = np.mean(l_accs)

mean_f1 = np.mean([r["weighted avg"]["f1-score"] for r in l_reports])

print(f"\tMean Acc: {mean_acc*100}% -- Mean Weighted Avg F1-Score: {mean_f1*100}%")



print("Polynomial SVM with Degree 3 Results:")

mean_acc = np.mean(p_accs)

mean_f1 = np.mean([r["weighted avg"]["f1-score"] for r in p_reports])

print(f"\tMean Acc: {mean_acc*100}% -- Mean Weighted Avg F1-Score: {mean_f1*100}%")
FEATURE_NAMES = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", "PAY_2",

                 "PAY_3", "PAY_4", "PAY_5", "PAY_6", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",

                 "BILL_AMT4", "BILL_AMT5", "BILLT_AMT6", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",

                 "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

# Due to computing time constraints use a much smaller training dataset here

x_train, y_train, x_test, y_test = trainTestSplit(dataset, 0.85)



def trainModel(x_train, y_train, x_test, y_test):

    svm = SVC(gamma='scale', kernel='poly', degree=3, 

              class_weight="balanced")

    svm.fit(x_train, y_train)

    return svm.score(x_test, y_test)



print(f"Will train on {x_train.shape[0]} samples and validate on {x_test.shape[0]} samples.")

# Train a baseline model for this data, including all the features

baseline_acc = trainModel(x_train, y_train, x_test, y_test)

print(f"Baseline Acc: {baseline_acc*100}%")

print("====================================")



remaining_features = np.arange(0, x_train.shape[1])

for i in range(x_train.shape[1]-1):

    feat_names = [FEATURE_NAMES[r] for r in remaining_features]

    print(f"Remaining features: ", end=' ')

    [print(feat, end=', ') for feat in feat_names]

    print()

    

    best_acc = 0.0

    least_impact_feature = 0

    # Find feature with least impact on performance

    for c in range(remaining_features.shape[0]):

        # Test by removing each of the columns

        curr_features = np.delete(remaining_features, c)

        part_x_train = x_train[:, curr_features]

        part_x_test = x_test[:, curr_features]

        

        acc = trainModel(part_x_train, y_train, part_x_test, y_test)

        if acc > best_acc:

            best_acc = acc

            least_impact_feature = c



    print(f"Removing feature {FEATURE_NAMES[remaining_features[least_impact_feature]]} -- Had the least impact on performance (Acc: {best_acc*100} %)")

    remaining_features = np.delete(remaining_features, least_impact_feature)

    print("====================================")

    

print(f"Last feature is {FEATURE_NAMES[remaining_features[0]]}")

part_x_train = x_train[:, remaining_features]

part_x_test = x_test[:, remaining_features]

acc = trainModel(part_x_train, y_train, part_x_test, y_test)

print(f"Accuracy: {acc*100} %")
# Get a dataset split for clustering and validation

x_train, y_train, x_test, y_test = trainTestSplit(dataset, 0.2)    



n_clusters = [2, 4, 8, 16, 32]

for n in n_clusters:

    print(f"K-Means with {n} clusters:")

    # Create K-Means model

    kmeans = KMeans(n_clusters=n)

    kmeans.fit(x_train) # Fit to training data

    # Get clusters from validation data

    clustered = kmeans.predict(x_test)

    # For each cluster, find the probability defaulting (that the client paid it next month)

    # by comparing the number of samples for each class

    highest_prob = 0.0

    highest_c = 0

    overall_acc = 0.0

    for c in range(n):

        # Retrieve samples that belong to the current cluster

        indices = np.where(clustered == c)[0]

        samples_in_cluster = [y_test[s] for s in indices]

        # How many of those samples are of customers with default paid next month?

        proportion = np.bincount(samples_in_cluster)

        prob = proportion[1] / np.sum(proportion)

        print(f"[Cluster {c}] - Probability of paid credit in this cluster: {prob*100} %")

        if prob > highest_prob:

            highest_c = c

            highest_prob = prob

            overall_acc = proportion[1] / y_test.shape[0]

    

    print(f"Choosing the single cluster {highest_c} as default_paid yields an overall Accuracy of {overall_acc*100} %")

    

    print()