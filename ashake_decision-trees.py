import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.model_selection import KFold

import operator

from math import log

from collections import Counter

from statistics import mean

from matplotlib import pyplot as plt
class Attribute(object):

    def __init__(self, label, vals):

        self.label = label

        self.vals = vals

    def __hash__(self):

        return hash(self.label)



    def __eq__(self, other):

        if not isinstance(other, type(self)): return NotImplemented

        return self.label == other.label



class Node(object):

    def __init__(self, label):

        self.label = label

        self.children = {}

        # Servono nel caso il nodo venisse potato

        self.pruned = False

        self.pruned_label = ""

    

    def __str__(self, level=0):

        rep = "("+repr(self.pruned_label or self.label)+")" if self.pruned or len(self.children) == 0 else self.label

        ret = "|\t"*level+rep+"\n"

        

        if not self.pruned:

            for child in self.children:

                ret += self.children[child].__str__(level+1)



        return ret
class DecisionTreeClassifier(object):

    def __init__(self, train_data, attributes):

        self.default = Counter(train_data['class']).most_common(1)[0][0]

        self.root = self.build_tree(train_data, attributes)



    def build_tree(self, train_data, attributes):

        

        # Calcolo le occorrenze delle varie classi

        class_occurs = Counter(train_data['class'])

        

        # Se non ho piu dati di trainig allora ritorno la label di default

        # che corrisponde alla la label di maggioranza negli esempi di training.

        if len(class_occurs) == 0:

            return Node(self.default)

        

        # 1° caso base: se gli esempi appartengono alla stessa classe allora costruisco un nodo foglia etichetta con la

        # classe.

        if len(class_occurs) == 1 :

            return Node(train_data['class'].iloc[0])

        

        # 2° caso base: se l'insieme degli attributi è vuoto allora ritorno un nodo foglia etichettato con la classe

        # di maggioranza negli esempi rimasti.

        if len(attributes) == 0:

            return Node(class_occurs.most_common(1)[0][0])

        

        # Seleziono l'attributo ottimo su cui fare il split

        best_attribute = self.__select_optimal_attribute(train_data, attributes)

        

        # Creo un nodo con etichettato con la label del attributo ottimo.

        node = Node(best_attribute.label)

        

        # Salvo l'eticheta di maggiornaza negli esempi nel caso il nodo venisse potato

        node.pruned_label = class_occurs.most_common(1)[0][0]

        

        # Rimuovo l'atttributo ottimo dall'insieme cosi da non considerarlo più negli livelli succesivi

        attributes.remove(best_attribute)

        

        #Ciclo su tutti i possibili valori dell'attributo ottimo per creare i sotto alberi

        for attr_val in best_attribute.vals:

            

            # Creo la partizione con le instanze aventi come valore per l'attributo best_attribute il valore attr_val

            partition = train_data.loc[train_data[best_attribute.label] == attr_val]

            # Costriusco il sotto albero corrispondente alla valore attr_val

            subtree = self.build_tree(partition, attributes.copy())

            

            node.children[attr_val] = subtree

        

        return node

    

    # Dati un insieme di esempi e attributi seleziona l'attiubuto che massimizza l'information gain

    def __select_optimal_attribute(self, train_data, attributes):

        

        # Calcola l'impurita pesata dell'insieme partizionato dal valore del attributo dato 

        def calc_weigthed_impurity (train_data, attribute, value):

            partition = train_data.loc[train_data[attribute] == value]

            return (len(partition.index)/len(train_data.index)) * self.__calc_impurity(partition)

        

        # Calcolo l'impurità attuale

        actual_impurity = self.__calc_impurity(train_data)

        

        #Calcolo l'information gain per ciascun attributo

        expected_inf_gain = {}

        for attr in attributes:

            # Calcolo l'information gain se partizionassi l'insieme usando i valori dell'attributo attr 

            inf_gain = actual_impurity - sum([calc_weigthed_impurity(train_data, attr.label, attr_val) for attr_val in attr.vals])            



            expected_inf_gain[attr] = inf_gain

        

        # Ritorno l'attributo che massimizza l'information gain

        return max(expected_inf_gain.items(), key = operator.itemgetter(1))[0]

    

    # Data un insieme di esempi ne calcola l'impurita a seconda della misure scelta

    def __calc_impurity(self, data):

        data_size = len(data.index)

        # calcolo le occorrenze delle varie classi

        class_occurs = Counter(data['class'])

    

        return  -sum([(class_occur/data_size) * log(class_occur/data_size) for _, class_occur in class_occurs.items()])

    

    # Classifica la data istanza

    def predict(self, instance):

        return self.__predict_impl(self.root, instance)

    

    # Esegue la classifcazione dato un albero di decisine e l'instanza da classificare

    def __predict_impl(self, node, instance):

        

        # Se ho raggiunto un nodo foglia restituisco l'etichetta associata

        if len(node.children) == 0:

            return node.label

        

        if not node.pruned:

            # Se il nodo non è potato allora proseguo lungo il ramo corrispondente al valore

            # dell'attributo assciato al nodo

            return self.__predict_impl(node.children[instance[node.label]], instance)

        else:

            # Altrimenti ritorno l'etichetta nel caso fosse potato

            return node.pruned_label

    

    # Esegue la potatura dell'albero di decisione

    def with_pruning(self, va_data):

        # Calcolo l'accuracy attuale

        actual_accuracy = accuracy_score(va_data['class'].to_numpy(), va_data.apply(self.predict, axis=1))

        # Seleziono il nodo ottimale da potare e l'accuracy attesa

        best_node, new_accuracy = self.__find_best_node(self.root, va_data, actual_accuracy)

        while best_node != None:

            # Imposto il nodo come potato

            best_node.pruned = True

            # Seleziono il successivo nodo ottimale da potare e l'accuracy attesa

            best_node, new_accuracy = self.__find_best_node(self.root, va_data, new_accuracy)

        return self

    

    # Seleziona il nodo ottimale per la potatura

    def __find_best_node(self, node, va_data, current_accuracy):

        # Se ho raggiunto una foglia o un nodo gia potato mi fermo

        if len(node.children) == 0 or node.pruned:

            return (None, 0.0)

        # Calcolo i potenziali nodi ottimo radicati negli sotto alberi del nodo corrente

        subtrees_best_nodes = [self.__find_best_node(subtree, va_data, current_accuracy) for _, subtree in node.children.items()]

        # Selezione il nodo milgiore negli sotto alberi

        best_node, exp_accuracy = max(subtrees_best_nodes, key = operator.itemgetter(1))

        # Imposto il nodo corrente come potato

        node.pruned = True

        # Calcolo l'accuracy dopo aver potato il nodo

        new_accuracy = accuracy_score(va_data['class'].to_numpy(), va_data.apply(self.predict, axis=1))

        # Reimposto il nodo a non potato cosi da non modificare l'albero

        node.pruned = False

        if exp_accuracy < new_accuracy and current_accuracy < new_accuracy:

            # Se potando il nodo ottengo un accuracy migliore allora

            # ritorno il nodo corrente con la nuova accuracy

            return (node, new_accuracy)

        else:

            # Altrimenti ritorno quello che ho trovato nei sotto alberi

            return (best_node, exp_accuracy)            
# Carico il dataset

data = pd.read_csv("../input/car-evaluation-data-set/car_evaluation.csv", encoding = 'utf-8',header = None)

# Rinomino le colonne

data.rename(columns = {0:'buying',1:'maintainence',2:'doors',3:'persons',4:'lug_boot',5:'safety',6:'class'},inplace = True)
# Costrusico l'insieme degli attributo con i loro possibili valori presi dal dataset.

attributes = set([Attribute(label, data[label].unique()) for label in data.columns])

attributes.remove(Attribute('class', None))



# Estraggo il validation set con un rapporto di 90:10

remaining_data, va_data = train_test_split(data, test_size=0.1, random_state=42)

# Costruisco il il kfold con k = 10

kf = KFold(n_splits = 10, shuffle = True, random_state = 42)



# Costriusco le dimensioni del trining set

step_size = len(remaining_data) // 20

train_sizes = range(10, len(remaining_data), step_size)

# Lista delle accuracy dei modelli appresi

unpruned_accuracies = []

pruned_accuracies = []



# Faccio apprendimento dei due modelli con training set sempre piu grandi partendo da 10 esempi

# fino ad usare l'intero trainig set del fold corrente

for size in train_sizes:

    unpruned_accuracies_i = []

    pruned_accuracies_i = []

    print('Number of Training Instances:', size)

    for train_ix, test_ix in kf.split(remaining_data):

        # Estraggo il training set del fold

        train_i = remaining_data.iloc[train_ix][:size]

        # Estraggo il test set del fold

        test_i = remaining_data.iloc[test_ix]

        

        # Costruisco il decision tree con e senza potatura

        unpruned_tree = DecisionTreeClassifier(train_i, attributes.copy())

        pruned_tree = DecisionTreeClassifier(train_i, attributes.copy()).with_pruning(va_data)

        

        # Estrahho le vere classi

        y_test = test_i['class'].to_numpy()

        

        # Faccio le predizioni usando i due alberi

        unpruned_preds = np.array(test_i.apply(unpruned_tree.predict, axis=1))

        pruned_preds = np.array(test_i.apply(pruned_tree.predict, axis=1))

        

        # Calcolo le accuracy dei due alberi

        unpruned_accuracies_i.append(accuracy_score(y_test, unpruned_preds))

        pruned_accuracies_i.append(accuracy_score(y_test, pruned_preds))

    

    print("Classification Accuracy for Pruned Tree:", mean(pruned_accuracies_i))

    print("Classification Accuracy for Unpruned Tree:", mean(unpruned_accuracies_i))

    print()

    # Prendo il valore medio delle accuracy calcolate nei fold

    unpruned_accuracies.append(mean(unpruned_accuracies_i))

    pruned_accuracies.append(mean(pruned_accuracies_i))
# Faccio il plot dell'andamento della performance con l'aumentare della dimensione del training set

plt.plot(train_sizes, pruned_accuracies, label='pruned tree')

plt.plot(train_sizes, unpruned_accuracies, label='unpruned tree')

plt.xlabel('Number of Training Instances')

plt.ylabel('Classification Accuracy on Test Instances')

plt.grid(True)

plt.title("Learning Curve")

plt.legend()

plt.show()
# Suddivido i dati rimanenti in training set e test set con un rapporto di 70:30

train_data, test_data = train_test_split(remaining_data, test_size=0.3, random_state=42)

# Costruisco il decision tree con e senza potatura

unpruned_tree = DecisionTreeClassifier(train_data, attributes.copy())

pruned_tree = DecisionTreeClassifier(train_data, attributes.copy()).with_pruning(va_data)



# Estrahho le vere classi

y_test = test_data['class'].to_numpy()

        

# Faccio le predizioni usando i due alberi

unpruned_preds = np.array(test_data.apply(unpruned_tree.predict, axis=1))

pruned_preds = np.array(test_data.apply(pruned_tree.predict, axis=1))



print("Unpruned DT Classification report:")

print(classification_report(y_test, unpruned_preds, target_names=test_i['class'].unique()))

print("Pruned DT Classification report:")

print(classification_report(y_test, pruned_preds, target_names=test_i['class'].unique()))