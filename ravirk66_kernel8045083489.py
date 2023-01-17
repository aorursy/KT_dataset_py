# Get all the required libraries
import sys
import csv
import argparse
import shlex
argString = '-l 54 -s  ../input/covtype.csv covtype_libsvm.txt'
import os
from sklearn import metrics
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.metrics import confusion_matrix
#Convert the covtype.csv into covtype_libsvm.txt (Libsvm Format)
def construct_line( label, line ):
	new_line = []
	if float( label ) == 0.0:
		label = "0"
	new_line.append( label )
	
	for i, item in enumerate( line ):
		if item == '' or float( item ) == 0.0:
			continue
		new_item = "%s:%s" % ( i + 1, item )
		new_line.append( new_item )
	new_line = " ".join( new_line )
	new_line += "\n"
	return new_line

# ---

parser = argparse.ArgumentParser()
parser.add_argument( "input_file", help = "path to the CSV input file" )
parser.add_argument( "output_file", help = "path to the output file" )

parser.add_argument( "-l", "--label-index", help = "zero based index for the label column. If there are no labels in the file, use -1.",
					 type = int, default = 0 )

parser.add_argument( "-s", "--skip-headers", help = "Use this switch if there are headers in the input file.", action = 'store_true' )

args = parser.parse_args(shlex.split(argString))
print(args)
#---	

i = open( args.input_file )
o = open( args.output_file, 'w' )

reader = csv.reader( i )
if args.skip_headers:
	headers = next(reader)

for line in reader:
	if args.label_index == -1:
		label = "1"
	else:
		label = line.pop( args.label_index )
	
	try:	
            new_line = construct_line( label, line )
            o.write( new_line )
	except ValueError:
		print ("Problem with the following line, skipping...")
		print (line)


#Load the libsvm file and obtain Train and Test Split
X, y = load_svmlight_file("covtype_libsvm.txt", n_features=55, dtype=int, zero_based=True)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y,
                                                random_state=0)
#Build a Random Forest Classifier Model
from sklearn.ensemble import RandomForestClassifier
#rf = RandomForestClassifier(n_estimators=55)
#model = RandomForestClassifier((n_estimators=600)
model = RandomForestClassifier(n_estimators=400)                               

# Fit the model
model.fit(Xtrain, ytrain)
#Predict usingthe Model
ypred = model.predict(Xtest)
print(metrics.classification_report(ypred, ytest))

mat = confusion_matrix(ytest, ypred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label');

metrics.accuracy_score(ytest,ypred)