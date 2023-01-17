# Imports

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from sklearn.cluster import SpectralClustering

from sklearn.metrics import pairwise_distances_argmin
# Read in the training data

train_data = pd.read_csv("../input/eval-lab-3-f464/train.csv")

train_data.head()
# Function to process the data

def process_data(data):

    data["gender"][data["gender"] == "Male"] = 1

    data["gender"][data["gender"] == "Female"] = 0

    

    data["Married"][data["Married"] == "Yes"] = 1

    data["Married"][data["Married"] == "No"] = 0

    

    data["Children"][data["Children"] == "Yes"] = 1

    data["Children"][data["Children"] == "No"] = 0

    

    data["TVConnection"][data["TVConnection"] == "DTH"] = 1

    data["TVConnection"][data["TVConnection"] == "Cable"] = 0.5

    data["TVConnection"][data["TVConnection"] == "No"] = 0

    

    for ch_no in range(1, 7):

        data["Channel"+str(ch_no)][data["Channel"+str(ch_no)] == "Yes"] = 1

        data["Channel"+str(ch_no)][data["Channel"+str(ch_no)] == "No"] = 0.5

        data["Channel"+str(ch_no)][data["Channel"+str(ch_no)] == "No tv connection"] = 0

    

    data["Internet"][data["Internet"] == "Yes"] = 1

    data["Internet"][data["Internet"] == "No"] = 0

    

    data["HighSpeed"][data["HighSpeed"] == "Yes"] = 1

    data["HighSpeed"][data["HighSpeed"] == "No"] = 0.5

    data["HighSpeed"][data["HighSpeed"] == "No internet"] = 0

    

    data["AddedServices"][data["AddedServices"] == "Yes"] = 1

    data["AddedServices"][data["AddedServices"] == "No"] = 0

    

    data["Subscription"][data["Subscription"] == "Monthly"] = 1

    data["Subscription"][data["Subscription"] == "Biannually"] = 0.5

    data["Subscription"][data["Subscription"] == "Annually"] = 0

    

    data["PaymentMethod"][data["PaymentMethod"] == "Cash"] = 1

    data["PaymentMethod"][data["PaymentMethod"] == "Bank transfer"] = 0.67

    data["PaymentMethod"][data["PaymentMethod"] == "Net Banking"] = 0.33

    data["PaymentMethod"][data["PaymentMethod"] == "Credit card"] = 0

    

    data["TotalCharges"][data["TotalCharges"] == " "] = -1

    

    for col in data.columns:

        data[col] = data[col].astype(np.float)

        data[col] -= data[col].mean()

        data[col] /= data[col].std()

        print("processed " + str(col))
# Process training data

train_data = train_data.drop(columns=["custId"])

process_data(train_data)

train_data["TotalCharges"][train_data["TotalCharges"] == -1] = train_data["TotalCharges"][train_data["TotalCharges"] != -1].mean()

train_data.head()
print(sum(train_data["Satisfied"] == 0.6011896708773844) / len(train_data))

print(sum(train_data["Satisfied"] == -1.6630311674919858) / len(train_data))
def run(n_clusters):

    # Train model

    model = KMeans(n_clusters=n_clusters)

    cluster_label = model.fit_predict(train_data.drop(columns=["Satisfied"]), train_data["Satisfied"])

    # Evaluate model and assign labels to clusters

    centroids, labels = {}, {}

    for c in range(n_clusters):

        cluster_size = len(train_data[cluster_label == c])

        frac_1 = sum(train_data["Satisfied"][cluster_label == c] == 0.6011896708773844) / cluster_size

        labels[c] = 1 if frac_1 >= 0.73 else 0

        centroids[c] = train_data.drop(columns=["Satisfied"])[cluster_label == c].mean()

        print("\t" + str(c) + 

              "(" + str(labels[c]) + ") : " + str(cluster_size) +

              " : " + str(frac_1))

    return centroids, labels
for n_cluster in range(2, 50):

    print(f"n_cluster : {n_cluster}")

    run(n_cluster)
# Train final model

n_clusters = 23

centroids, labels = run(n_clusters)

print(centroids)
# Read in the test data

test_data = pd.read_csv("../input/eval-lab-3-f464/test.csv")

test_data.head()
# Process the test data

test_custId = test_data["custId"]

test_data = test_data.drop(columns=["custId"])

process_data(test_data)

test_data["TotalCharges"][test_data["TotalCharges"] == -1] = test_data["TotalCharges"][test_data["TotalCharges"] != -1].mean()

test_data.head()
# Make predictions

test_clusters = pairwise_distances_argmin(test_data, np.array(list(centroids.values())))

submission = pd.DataFrame(columns=["custId", "Satisfied"])

submission["custId"] = test_custId

submission["Satisfied"] = [labels[label] for label in test_clusters]

submission.head(20)

submission.to_csv("submission.csv", index=False)

# print(sum(submission["Satisfied"] == 0))

# print(sum(submission["Satisfied"] == 1))