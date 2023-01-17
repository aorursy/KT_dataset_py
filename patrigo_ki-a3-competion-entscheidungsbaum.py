# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Read some arff data
def read_data(filename):
    f = open(filename)
    data_line = False
    data = []
    for l in f:
        l = l.strip() # get rid of newline at the end of each input line
        if data_line:
            content = [float(x) for x in l.split(',')]
            if len(content) == 3:
                data.append(content)
        else:
            if l.startswith('@DATA'):
                data_line = True
    return data

d = read_data("../input/kiwhs-comp-1-complete/train.arff")
train, val =train_test_split(d, test_size=0.25) # 25% der Daten werden zu Validierungsdatensätzen
print("Trainingssätze:",len(train))
print("Validierungssätze:",len(val))
def get_key_X(item):
    return item[0]
def get_key_Y(item):
    return item[1]

def get_gini(left_A,left_B,right_A,right_B):
    left_gini = 0
    right_gini = 0
    if left_A + left_B > 0:
        left_p_A = (left_A / (left_A + left_B)) ** 2
        left_p_B = (left_B / (left_A + left_B)) ** 2
        left_gini = 1 - left_p_A - left_p_B
    if right_A + right_B > 0:
        right_p_A = (right_A / (right_A + right_B)) ** 2
        right_p_B = (right_B / (right_A + right_B)) ** 2
        right_gini = 1 - right_p_A - right_p_B
    return left_gini + right_gini

def get_gini_single(node):
    amount_A = sum(c == 1  for [x,y,c] in node)
    amount_B = sum(c == -1 for [x,y,c] in node)
    
    result = []
    p_A = (amount_A / len(node))**2
    p_B = (amount_B / len(node))**2
    result.append(1-p_A-p_B)
    
    if amount_A >= amount_B:
        result.append(1)
    else:
        result.append(-1)
    return result # Format: [Gini-Index, Meist vorhandene Klasse]
max_layers = 3
purity_threshold = 0.15

mengen_baum = []
desc_tree = [[]] 
# Format: [Variable,Wert]
# Ist Variable = -1 so ist dies ein Blattknoten und enthält die Klassifizierung als Wert; sonst 0 => x, 1 => y; Wert Splittpunkt

mengen_baum.append(train)
i = 0
best_gini_X = 1
best_gini_Y = 1
gini_stelle_X = 0
gini_stelle_Y = 0

max_nodes = 0
l = 0
while l <= max_layers:
    max_nodes = max_nodes + (2 ** l)
    l = l + 1


while i < max_nodes:
    if desc_tree[i] == []:
        x_sorted = sorted(mengen_baum[i], key=get_key_X)
        y_sorted = sorted(mengen_baum[i], key=get_key_Y)
        
        #Finde besten Splitpunkt für einen Split, auf der X-Achse
        
        left_A = 0 # Anzahl Klasse 1, linker Zweig
        left_B = 0 # Anzahl Klasse -1, linker Zweig
        right_A = sum(c == 1  for [x,y,c] in x_sorted) # Anzahl Klasse 1, rechter Zweig
        right_B = sum(c == -1 for [x,y,c] in x_sorted) # Anzahl Klasse -1, rechter Zweig
              
        k = 1
        last_x = None

        while k < len(x_sorted) - 1:
            if x_sorted[k-1][2] == 1:
                left_A = left_A + 1
                right_A = right_A - 1
            else:
                left_B = left_B + 1
                right_B = right_B - 1

            if x_sorted[k][0] != last_x or k == len(x_sorted):
                last_x = x_sorted[k][0]
                gini_score = get_gini(left_A,left_B,right_A,right_B)
                if gini_score < best_gini_X:
                    best_gini_X = gini_score
                    gini_stelle_X = x_sorted[k][0]
            k = k + 1


        # Best Gini für die Y-Achse bestimmen, wie zuvor mit X

        left_A = 0 # Anzahl Klasse 1, linker Zweig
        left_B = 0 # Anzahl Klasse -1, linker Zweig
        right_A = sum(c == 1  for [x,y,c] in y_sorted) # Anzahl Klasse 1, rechter Zweig
        right_B = sum(c == -1 for [x,y,c] in y_sorted) # Anzahl Klasse -1, rechter Zweig
        k = 0
        
        last_y = None
        
        while k < len(y_sorted) - 1:
            if y_sorted[k-1][2] == 1:
                left_A = left_A + 1
                right_A = right_A - 1
            else:
                left_B = left_B + 1
                right_B = right_B - 1

            if y_sorted[k][1] != last_y or k == len(y_sorted) - 1:
                last_y = y_sorted[k][1]
                gini_score = get_gini(left_A,left_B,right_A,right_B)
                if gini_score < best_gini_Y:
                    best_gini_Y = gini_score
                    gini_stelle_Y = y_sorted[k][1]
            k = k + 1
        
        #--------------------------------------------------------------------------------------
        
        # Erstelle die neuen Zweige
        linker_zweig = []
        rechter_zweig = []
        
        n = 1
        
        if best_gini_X <= best_gini_Y: # Enscheide Welcher Gini besser ist
            # Fall x-Gini besser
            desc_tree[i] = [0,gini_stelle_X]
            linker_zweig.append(x_sorted[0])
            while n < len(x_sorted) - 1: # Sortiere die Punkte in die neuen Zweige
                if x_sorted[n][0] <= gini_stelle_X:
                    linker_zweig.append(x_sorted[n])
                else:
                    rechter_zweig.append(x_sorted[n])
                n = n + 1
            rechter_zweig.append(x_sorted[len(x_sorted) - 1])
        else: # Fall y-Gini besser
            desc_tree[i] = [1,gini_stelle_Y]
            linker_zweig.append(y_sorted[0])
            while n < len(y_sorted) - 1: # Sortiere die Punkte in die neuen Zweige
                if y_sorted[n][1] <= gini_stelle_Y:
                    linker_zweig.append(y_sorted[n])
                else:
                    rechter_zweig.append(y_sorted[n])
                n = n + 1
            rechter_zweig.append(y_sorted[len(y_sorted) - 1])
        mengen_baum.append(linker_zweig)
        mengen_baum.append(rechter_zweig)

        #------------------------------------------------------------------------------------------
        
        #Prüfe ob die Mengen der neu erstelleten Zweige rein genug sind, falls Ja füge eine Klassifizierung zum Entscheidungsbaum hinzu 
        
        if len(linker_zweig) > 0:
            gini_left = get_gini_single(linker_zweig)
            if gini_left[0] <= purity_threshold:
                desc_tree.append([-1, gini_left[1]])
            else:
                desc_tree.append([])
        else:
            desc_tree.append([-2])

        if len(rechter_zweig) > 0:
            gini_right = get_gini_single(rechter_zweig)
            if gini_right[0] <= purity_threshold:
                desc_tree.append([-1, gini_left[1]])
            else:
                desc_tree.append([])
        else:
            desc_tree.append([-2])
            
    else:
        # Sollte an dieser Stelle bereits für die Menge des i-ten Elements eine Klassifiezierung erfolgt sein,
        # füge diese als neue Blätter hinzu, sowie die Mengen zum Erstellen des Baums.
        # Dies ist wichtig, damit der Baum an den richtigen Inidices gelesen wird bei der Erstellung und
        # dass der Entscheidungsbaum später richtig durchwandert wird.
        desc_tree.append(desc_tree[i])
        mengen_baum.append(mengen_baum[i])
        desc_tree.append(desc_tree[i])
        mengen_baum.append(mengen_baum[i])
    i = i + 1
    
# Wenn die maximale Anzahl an Layern erreicht wurde, forme die verbleibenden Blätter die keine Klassifizierungen sind in solche um
while i < len(desc_tree):
    if desc_tree[i] == []:
        A = sum(c == 1  for [x,y,c] in mengen_baum[i])
        B = sum(c == -1 for [x,y,c] in mengen_baum[i])
        if A >= B:
            desc_tree[i] = [-1, 1]
        else:
            desc_tree[i] = [-1,-1]
    i = i + 1
print("Entscheidungsbaum: (Kinder des Knotens i sind i * 2 + 1 und i * 2 + 2)")
print()
print(desc_tree)
def predict(item):
    k = 0
    while k < len(desc_tree) and desc_tree[k][0] != -1:
        if item[desc_tree[k][0]] <= desc_tree[k][1]:
            k = k * 2 + 1
        else:
            k = k * 2 + 2
    return desc_tree[k][1]

guesses = []
for x in val:
    item = [x[0],x[1]]
    guesses.append(predict(item))
correct_guesses = 0
for i in range(0,len(guesses)):
    if guesses[i] == val[i][2]:
        correct_guesses = correct_guesses + 1
print("Richtig Kategorisiert: %d/%d"%(correct_guesses,len(guesses)))
test_data = pd.read_csv('../input/kiwhs-comp-1-complete/test.csv')
test_data = pd.DataFrame(test_data)
test_data = test_data.drop(labels = ["Id"], axis=1)
test_data = test_data.values
results = []
i = 0
while i < len(test_data):
    results.append([i,predict(test_data[i])])
    i = i + 1
    
frame = pd.DataFrame(results)
frame.columns = ["Id (String)", "Category (String)"]
frame.to_csv("submission.csv",index=False)