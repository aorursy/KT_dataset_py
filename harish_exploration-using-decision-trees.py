# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import tree

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Read data
data = pd.read_csv("../input/Admission_Predict.csv")
# Add new column high_admission_chance
high_admission_chance_threshold = .75
data['high_admission_chance'] = (data['Chance of Admit '] > high_admission_chance_threshold).astype(int)


# Create Decision tree to predict admission chance based on input variables
x_columns = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ','CGPA','Research']
y_column = 'high_admission_chance'

# To avoid overfitting ensure that each region/ leaf node has at least 20 samples
classifier = tree.DecisionTreeClassifier(min_samples_leaf=20)
classifier.fit(data[x_columns], data[y_column])

# Render Decision Tree
import graphviz
dot_data = tree.export_graphviz(classifier, out_file=None) 
graph = graphviz.Source(dot_data)
graph
# Create helper method to extract all leaf nodes and conditions from decision tree classifier
def get_regions(decision_tree_classifier, cols):
    tree = decision_tree_classifier.tree_
    regions = []
    def traverse_tree(id, conditions):
        left_child = tree.children_left[id]
        right_child = tree.children_right[id]
        if left_child < 0 and right_child < 0:
            region = {}
            region['condition'] = conditions.copy()
            region['class_values'] = tree.value[id][0]
            regions.append(region)
        if left_child >=0:
            conditions.append((cols[tree.feature[id]], tree.threshold[id], True))
            traverse_tree(left_child, conditions)
            del conditions[-1]
        if right_child >=0:
            conditions.append((cols[tree.feature[id]], tree.threshold[id], False))
            traverse_tree(right_child, conditions)
            del conditions[-1]
            
    traverse_tree(0, [])
    return regions
# Get all regions
regions = get_regions(classifier, x_columns)
# Helper method to gte summary stats for samples of a region. Samples are identified using conditions associated 
# with the region/ leaf node
def get_summary_stats(data, region, columns, y_column):
    all_mask = None
    for (attr, threshold, lessThan) in region['condition']:        
        if lessThan:
            mask = data[attr] <= threshold
        else:
            mask = data[attr] > threshold
            
        if all_mask is None:
            all_mask = mask
        else:
            all_mask = all_mask & mask
    region_data = data[all_mask]
    return region_data.groupby(y_column).agg(['min', 'mean', 'median', 'max'])
# Helper method to print detailed stats of a region
def print_region_stats(region_no, region, region_summary_stats, columns, class_labels):
    
    def get_condition_string(region):
        all_condition_str = None
        for (attr, threshold, lessThan) in region['condition']:
            condition_str = "{0} {1} {2}".format(attr, "<=" if lessThan else ">", threshold)
            if all_condition_str is None:
                all_condition_str = condition_str
            else:
                all_condition_str = "{0} and {1}".format(all_condition_str, condition_str)
        return all_condition_str
    
    class_counts_str = ""
    for class_label, class_value in zip(class_labels, region['class_values']):
            class_counts_str += "{0}={1} ".format(class_label, int(class_value))  
    
    print("Region # {0}: {1}".format(region_no, class_counts_str))
    print("\tCondition: {0}".format(get_condition_string(region)))
    
    target_class_distribution_value = ""
    for class_label, class_value in zip(class_labels, region['class_values']):
            target_class_distribution_value += "{0}={1:.2f}% ".format(class_label, ((class_value*100)/sum(region['class_values'])))   
    print("\tClass Probablity: {0}".format(target_class_distribution_value))

    for index in region_summary_stats.index:
        print("\tSummary Statistics [{0}] (min, avg, median, max)".format(class_labels[index]))
        for col in columns:
            print("\t\t{0:20}: {1:8.2f} {2:8.2f} {3:8.2f} {4:8.2f}".format(col, 
                                                     region_summary_stats.loc[index][col]['min'], 
                                                     region_summary_stats.loc[index][col]['mean'], 
                                                     region_summary_stats.loc[index][col]['median'], 
                                                     region_summary_stats.loc[index][col]['max']))
    print("")
# Print stats for each region
for index in range(0, len(regions)):
    region_stat = get_summary_stats(data, regions[index], x_columns, y_column)
    print_region_stats(index, regions[index], region_stat, x_columns, ['Not High Chance of Adm.', 'High Chance of Adm.'])