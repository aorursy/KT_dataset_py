# Im Folgenden benötigte Pakete

import numpy as np

import matplotlib.pyplot as plt

from prettytable import PrettyTable as pt

import random as rdm



# Laut Aufgabe zu berücksichtigendes Intervall

min_x_value = 0.3

max_x_value = 4.0



# Gibt das arithmetische Mittel zurück.

def avg_value(values):

    sum = 0.0

    for value in values:

        sum += value

    return sum / len(values)



# Zeichnet Daten in ein Koordinatensystem

def draw_data(title, x, y):

    plt.title(title)

    plt.xlabel("Fernsehzeit")

    plt.ylabel("Dauer Tiefschlaf")

    plt.scatter(x,y)

    plt.show()
# Klasse LinearFunction - hier der Schätzer

class LinearFunction:

    

    # anzupassende Variablen in linearer Funktion

    m_value = 0.0

    b_value = 0.0

    

    # zulässiges Intervall für x-Werte

    min_x = 0.0

    max_x = 0.0

    

    # Konstruktor für Lineare Funktion

    def __init__(self, min_x, max_x):

        self.min_x = min_x

        self.max_x = max_x

       

    # Grafische Ausgabe der Linearen Funktion

    def to_graph(self):

        # Gerade zwischen kleinstem und größtem x-Wert

        # (des gegebenen Intervalls) zeichnen

        line = plt.plot([self.min_x, self.max_x],

                 [self.guess(self.min_x), self.guess(self.max_x)])

        plt.setp(line, color='r')

        

    # b- & m-Wert bestimmen, für die die Abstände aller Punkte

    # zur Geraden/Funktion minimal sind. Dazu werden spezielle

    # Formeln zum errechnen von b und m genutzt

    def train_ordinary_least_squares(self, x, y):

       # pointColl = PointCollection(data)

        avg_x = avg_value(x)

        avg_y = avg_value(y)

        

        # m_value bestimmen

        # Zähler des Ausdrucks bestimmen

        sum_numerator = 0.0

        for i in range(len(x)):

            val_x = x[i]

            val_y = y[i]

            sum_numerator += (val_x-avg_x) * (val_y-avg_y)

            

        # Nenner des Ausdrucks bestimmen

        sum_denominator = 0.0

        for i in range(len(x)):

            val_x = x[i]

            sum_denominator += np.square(val_x-avg_x)

            

        # Ergebnis des Ausdrucks (m_value) bestimmen

        self.m_value = sum_numerator / sum_denominator

        print("m_value: %.4f" % self.m_value)

        

        # Ausgehend vom m_value b_value bestimmen

        self.b_value = avg_y - self.m_value * avg_x

        print("b_value: %.4f" % self.b_value)    

        

    # ALTERNATIVE: Ermittle die b- & m-Werte mittels Gradient Descent

    def train_stochastic_gradient_descent(self, x, y):

        learning_rate = 0.1

        precision = 0.0001

        

        max_rounds, stop = 100, False

        last_round_m_value, last_round_b_value = 0.0, 0.0

        for counter in range(max_rounds):

            

            # Alle Datensätze (Punkte) durchgehen

            for i in range(len(x)):

                guess = self.guess(x[i])

                # entstandenen Fehler bei der Vorhersage bestimmen

                error = y[i] - guess

                # b und m anhand des error anpassen

                self.m_value += error * x[i] * learning_rate

                self.b_value += error * learning_rate  

                

            # Wenn auf der Ebene der gewünschten Genauigkeit keine Änderung

            # mehr stattfindet: Training beenden

            if abs(last_round_m_value - self.m_value) < precision and abs(last_round_b_value - self.b_value) < precision:

                break                

            last_round_m_value = self.m_value

            last_round_b_value = self.b_value

            

        print("m_value: %.4f" % self.m_value)

        print("b_value: %.4f" % self.b_value) 

        

    # Vorhersage des y-Werts zum gegebenen x-Wert (=Funktion ausrechnen)

    def guess(self, x_value):

        return self.m_value * x_value + self.b_value
# Trainingsdaten (jeweils x- und y-Wert)

data = np.array([ [0.3, 5.8], [2.2, 4.4], [0.5, 6.5], [0.7, 5.8],

    [1.0, 5.6], [1.8, 5.0], [3.0, 4.8], [0.2, 6.0], [2.3, 6.1]])



# Nach x-Wert sortieren

data = data[data[:,0].argsort()]



# Alle Werte an Position 0 in Tupeln (=Fernsehzeit)

x = data[:,0]

# Alle Werte an Position 1 in Tupeln (=Dauer Tiefschlaf)

y = data[:,1]



# Grafische Anzeige in Koordinatensystem

draw_data("Trainingsdaten", x, y)
linFunc_ols = LinearFunction(min_x_value, max_x_value)

linFunc_ols.train_ordinary_least_squares(x, y)
linFunc_ols.to_graph()

draw_data("Trainingsergebnis", x, y)
linFunc_sgd = LinearFunction(min_x_value, max_x_value)

linFunc_sgd.train_stochastic_gradient_descent(x, y)
linFunc_sgd.to_graph()

draw_data("Trainingsergebnis", x, y)
def make_guesses(linFuncs):

    # Listen (=Spalten) der x- und y-Werte zur späteren Darstellung in Tabelle

    x_values = []

    y_values = {}



    # x-Startwert sei hier 0.0

    x_value = 0.0



    # Solange 0.5 addieren bis der x-Wert über dem minimalen x-Wert liegt

    while x_value < min_x_value:

        x_value += 0.5

        

    # Alle x-Werte ermitteln

    while x_value <= max_x_value:

        x_values.append("%.1fh" % x_value)

        x_value += 0.5

 

    # Ausgabetabelle erstellen

    result_table = pt()

    result_table.add_column("Fernsehzeit", x_values)

    

    # Für jeden x-Wert, der unterhalb des maximalen x-Werts liegt wird

    # eine Schätzung des y-Werts vorgenommen.

    for key, value in linFuncs.items():

        y_values[key] = []

        for x in x_values:

            x = float(x[:len(x)-1])

            y_value = value.guess(x)

            y_values[key].append("%.2fh" % y_value)        

        result_table.add_column("[%s] Tiefschlafzeit" % key, y_values[key])

    

    print(result_table)
make_guesses({'OLS': linFunc_ols, 'ALTERNATIVE': linFunc_sgd})