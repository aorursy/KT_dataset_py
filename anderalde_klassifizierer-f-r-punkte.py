import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import csv
import os
import math
print(os.listdir("../input/"))


# Any results you write to the current directory are saved as output.

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

#d = read_data("../input/trainingsdaten-reupload/train.arff")
d = read_data("../input/neue-trainingsdaten/train-skewed.arff")

l = len(d)
indices = range(l)
#print(indices)
#print(list(indices))

validation = np.random.choice(l,0,replace = False)
#print(validation)

train = set(indices) - set(validation)
#print(train)


        
def give_color_for_X(x_value): #Liefert zu einem konkreten X-Wert eines Punktes aus den Trainingsdaten seine Farbe.
    result = 0
    for x in d:
        if x[0] == x_value:
            result = x[2]
    return result

def give_color_for_Y(y_value): #Liefert zu einem konkreten Y-Wert eines Punktes aus den Trainingsdaten seine Farbe.
    result = 0
    for y in d:
        if y[1] == y_value:
            result = y[2]
    return result
        
def give_subsets(given_set,split_value,is_X_value): #Teilt das Set der Indizes in 2 Subsets, anhand eines vorher bestimmten optimalen Splitwertes.
    set_two = set()
    if is_X_value:
        column = 0
    else:
        column = 1
    for x in given_set:
        if d[x][column] > split_value:
            set_two.add(x)
    set_one = given_set - set_two
    results = []
    results.append(set_one)
    results.append(set_two)
    
    #print(set_one)
    #print(set_two)
    return results
    
def divide_data_X_scale(data_indices): #Teilt die Daten einmal anhand eines x-Wertes. liefert den besten x-Wert zum teilen zurück, und die Farbe, die sich rechts davon befindet.
    x_values = []
    for x in data_indices:
        x_values.append(d[x][0])
    x_values.sort()
    #print(x_values)
    
    #-3,9 bis 4.6 
    #sucht jetzt beste Aufteilung
    
    #Startaufteilung
    blue_counter_left = 0
    red_counter_left = 0
    blue_counter_right = 0
    red_counter_right = 0
    for x in range(0, 400):
        if x in data_indices:
            if d[x][2] == -1:
                blue_counter_right += 1
            else:
                red_counter_right += 1
    blue_tendency = blue_counter_right - blue_counter_left #Positive Werte heißen: Tendenz nach rechts, Negative Werte heißen: Tendenz nach links
    red_tendency = red_counter_right - red_counter_left
    division_quality = abs(blue_tendency - red_tendency)
    best_division_quality = division_quality
    best_blue_tendency = blue_tendency
    best_red_tendency = red_tendency
    best_division_value = 0
    print()
    print('Start:')
    print('Blaue links: ' + str(blue_counter_left) + ' / Rote links: ' + str(red_counter_left) + ' / Blaue rechts: ' + str(blue_counter_right) + ' / Rote rechts: ' + str(red_counter_right))
    #print('Blaue Tendenz: ' + str(blue_tendency) + ' / Rote Tendenz: ' + str(red_tendency))
    #print('Einteilungsqualität: ' + str(division_quality))
    
    #Vergleiche andere Aufteilungen
    
    for x in x_values:
        color = give_color_for_X(x)
        if color == -1:
            blue_counter_right -= 1
            blue_counter_left += 1
        else:
            red_counter_right -= 1
            red_counter_left += 1
        blue_tendency = blue_counter_right - blue_counter_left
        red_tendency = red_counter_right - red_counter_left
        division_quality = abs(blue_tendency - red_tendency)
        
        #print(division_quality)
        if division_quality > best_division_quality:
            best_division_quality = division_quality
            best_blue_tendency = blue_tendency
            best_red_tendency = red_tendency
            best_division_value = x
    print()
    print('Beste Einteilung:')
    print('Blaue Tendenz: ' + str(best_blue_tendency) + ' / Rote Tendenz: ' + str(best_red_tendency))
    print('Einteilungsqualität: ' + str(best_division_quality))
    print('Wert zum Einteilen: ' + str(best_division_value))
    
    right_color = 'white'
    if best_blue_tendency - best_red_tendency >= 0:
        right_color = 'blue'
    else:
        right_color = 'red'
    
    results = []
    results.append(best_division_value)
    results.append(right_color)
    
    return results
    
def divide_data_Y_scale(data_indices): #Teilt die Daten einmal anhand eines y-Wertes. liefert den besten y-Wert zum teilen zurück, und die Farbe, die sich oberhalb davon befindet.
    y_values = []
    for y in data_indices:
        y_values.append(d[y][1])
    y_values.sort()
    #print(y_values)
    
    #sucht jetzt beste Aufteilung
    
    #Startaufteilung
    blue_counter_bot = 0
    red_counter_bot = 0
    blue_counter_top = 0
    red_counter_top = 0
    for y in range(0, 400):
        if y in data_indices:
            if d[y][2] == -1:
                blue_counter_top += 1
            else:
                red_counter_top += 1
    blue_tendency = blue_counter_top - blue_counter_bot #Positive Werte heißen: Tendenz nach oben, Negative Werte heißen: Tendenz nach unten
    red_tendency = red_counter_top - red_counter_bot
    division_quality = abs(blue_tendency - red_tendency)
    best_division_quality = division_quality
    best_blue_tendency = blue_tendency
    best_red_tendency = red_tendency
    best_division_value = 0
    print()
    print('Start:')
    print('Blaue unten: ' + str(blue_counter_bot) + ' / Rote unten: ' + str(red_counter_bot) + ' / Blaue oben: ' + str(blue_counter_top) + ' / Rote oben: ' + str(red_counter_top))
    
    #Vergleiche andere Aufteilungen
    
    for y in y_values:
        color = give_color_for_Y(y)
        if color == -1:
            blue_counter_top -= 1
            blue_counter_bot += 1
        else:
            red_counter_top -= 1
            red_counter_bot += 1
        blue_tendency = blue_counter_top - blue_counter_bot
        red_tendency = red_counter_top - red_counter_bot
        division_quality = abs(blue_tendency - red_tendency)
        
        #print(division_quality)
        if division_quality > best_division_quality:
            best_division_quality = division_quality
            best_blue_tendency = blue_tendency
            best_red_tendency = red_tendency
            best_division_value = y
    print()
    print('Beste Einteilung:')
    print('Blaue Tendenz: ' + str(best_blue_tendency) + ' / Rote Tendenz: ' + str(best_red_tendency))
    print('Einteilungsqualität: ' + str(best_division_quality))
    print('Wert zum Einteilen: ' + str(best_division_value))
    
    right_color = 'white'
    if best_blue_tendency - best_red_tendency >= 0:
        top_color = 'blue'
    else:
        top_color = 'red'
    
    results = []
    results.append(best_division_value)
    results.append(top_color)
    
    return results

#Liest die Testdaten ein.

test_data = pd.read_csv("../input/ki-whs-a3/test.csv")
test_data = pd.DataFrame(test_data)
test_data = test_data.drop(labels = ["Id"], axis=1)
test_data_X = test_data.drop(labels = ["Y"], axis=1)
test_data_Y = test_data.drop(labels = ["X"], axis=1)
test_data_X = test_data_X.values
test_data_Y = test_data_Y.values
results = []
i = 0

#Bestimmt die besten Splitwerte basierend auf den Trainingsdaten.

first_division = divide_data_X_scale(train)
subsets = give_subsets(train,first_division[0],True)
second_division_left = divide_data_Y_scale(subsets[0])
second_division_right = divide_data_Y_scale(subsets[1])

#Weist den Testdaten die Farben zu, basierend auf den vorher bestimmten Splitwerten.

while i < len(test_data):
    #print(test_data[i])
    if test_data_X[i] < first_division[0]:
        if test_data_Y[i] < second_division_left[0]:
            if second_division_left[1] == 'red':
                color = -1
            else:
                color = 1
        else:
            if second_division_left[1] == 'red':
                color = 1
            else:
                color = -1
    else:
        if test_data_Y[i] < second_division_right[0]:
            if second_division_right[1] == 'red':
                color = -1
            else:
                color = 1
        else:
            if second_division_right[1] == 'red':
                color = 1
            else:
                color = -1
    results.append([i,color])
    i = i + 1
    
frame = pd.DataFrame(results)
frame.columns = ["Id (String)", "Category (String)"]
frame.to_csv("submission.csv",index=False)
