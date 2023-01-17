import numpy as np



 

cm  = np.array([[ 2868.0,  9.0, 74.0, 0.0],

                [  8.0,  95.0,  7.0, 16.0],

                [  13.0,  0.0, 666.0,  14.0],

                [  10.0,  25.0,  55.0, 2008.0]])
def precision(label, confusion_matrix):

    col = confusion_matrix[:, label]

    return confusion_matrix[label, label] / col.sum()

 

def recall(label, confusion_matrix):

    row = confusion_matrix[label, :]

    return confusion_matrix[label, label] / row.sum()

 

def precision_macro_average(confusion_matrix):

    rows, columns = confusion_matrix.shape

    sum_of_precisions = 0

    for label in range(rows):

        sum_of_precisions += precision(label, confusion_matrix)

    return sum_of_precisions / rows

 

def recall_macro_average(confusion_matrix):

    rows, columns = confusion_matrix.shape

    sum_of_recalls = 0

    for label in range(columns):

        sum_of_recalls += recall(label, confusion_matrix)

    return sum_of_recalls / columns



 

print("label  recall  precision")

for label in range(4):

    print(f"{label:5d} {precision(label, cm):9.3f} {recall(label, cm):6.3f}")

    

print("precision total:", precision_macro_average(cm))

print("recall total:", recall_macro_average(cm))

 

def accuracy(confusion_matrix):

    diagonal_sum = confusion_matrix.trace()

    sum_of_all_elements = confusion_matrix.sum()

    return (diagonal_sum / sum_of_all_elements)



accuracy(cm)



print("F1",(2*(precision_macro_average(cm) * recall_macro_average(cm)) / (precision_macro_average(cm) + recall_macro_average(cm)))*100)

print("Accuracy_percentage",(accuracy(cm))*100)

print("Misclass_percentage",100-(accuracy(cm)*100))
