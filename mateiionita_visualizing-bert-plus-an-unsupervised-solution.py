# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



from sklearn.metrics import log_loss

import matplotlib.pyplot as plt

% matplotlib inline
emb_size = 768



def parse_json(embeddings):

	'''

	Parses the embeddigns given by BERT, and suitably formats them to be passed to the MLP model



	Input: embeddings, a DataFrame containing contextual embeddings from BERT, as well as the labels for the classification problem

	columns: "emb_A": contextual embedding for the word A

	         "emb_B": contextual embedding for the word B

	         "emb_P": contextual embedding for the pronoun

	         "label": the answer to the coreference problem: "A", "B" or "NEITHER"



	Output: X, a numpy array containing, for each line in the GAP file, the concatenation of the embeddings of the target words

	        Y, a numpy array containing, for each line in the GAP file, the one-hot encoded answer to the coreference problem

	'''

	embeddings.sort_index(inplace = True) # Sorting the DataFrame, because reading from the json file messed with the order

	X = np.zeros((len(embeddings),3*emb_size))

	Y = np.zeros((len(embeddings), 3))



	# Concatenate features

	for i in range(len(embeddings)):

		A = np.array(embeddings.loc[i,"emb_A"])

		B = np.array(embeddings.loc[i,"emb_B"])

		P = np.array(embeddings.loc[i,"emb_P"])

		X[i] = np.concatenate((A,B,P))



	# One-hot encoding for labels

	for i in range(len(embeddings)):

		label = embeddings.loc[i,"label"]

		if label == "A":

			Y[i,0] = 1

		elif label == "B":

			Y[i,1] = 1

		else:

			Y[i,2] = 1



	return X, Y
development = pd.read_json("../input/taming-the-bert-a-baseline/contextual_embeddings_gap_development.json")

X_development, Y_development = parse_json(development)



# There may be a few NaN values, where the offset of a target word is greater than the max_seq_length of BERT.

# They are very few, so rather than dealing with the problem, I'm replacing those rows with random values.

remove_development = [row for row in range(len(X_development)) if np.sum(np.isnan(X_development[row]))]

X_development[remove_development] = np.random.randn(3*emb_size)
n_rows = len(X_development)

index = list(range(n_rows))

distance = pd.DataFrame(index = index, columns = ["d_PA", "d_PB", "d_AB"])

for i in index:

	distance.loc[i,"d_PA"] = np.linalg.norm(X_development[i,2*emb_size:] - X_development[i,:emb_size], ord = 2)  

	distance.loc[i,"d_PB"] = np.linalg.norm(X_development[i,emb_size:2*emb_size] - X_development[i,2*emb_size:], ord = 2) 

	distance.loc[i,"d_AB"] = np.linalg.norm(X_development[i,:emb_size] - X_development[i,emb_size:2*emb_size], ord = 2)
plt.scatter(distance["d_PA"] / distance["d_PB"], 

            distance["d_AB"] / (distance["d_PB"]+ distance["d_PA"]), 

            c = np.argmax(Y_development[:n_rows], axis = 1), 

            alpha = 0.5)

plt.xlabel("d(A,Pronoun) / d(B,Pronoun)")

plt.ylabel("d(A,B) / d(A,Pronoun) + d(B,Pronoun)")

plt.colorbar()

plt.show()
threshold = -1



def softmax(v):

	exp = np.exp(v)

	return exp / np.sum(exp, axis = 1, keepdims = True)



distance["A/B"] = 1- distance["d_PA"] / distance["d_PB"] 

distance["B/A"] = 1- distance["d_PB"] / distance["d_PA"]  

distance["N/A+B"] = threshold - distance["d_AB"] / (distance["d_PB"]+ distance["d_PA"])

values = distance[["A/B", "B/A", "N/A+B"]].values.astype(float)

prediction = softmax(values)

print("The score is :", log_loss(Y_development, prediction))
submission = pd.read_csv("../input/gendered-pronoun-resolution/sample_submission_stage_1.csv", index_col = "ID")

submission["A"] = prediction[:,0]

submission["B"] = prediction[:,1]

submission["NEITHER"] = prediction[:,2]

submission.to_csv("submission_bert_more_data.csv")