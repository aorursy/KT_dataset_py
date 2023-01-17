import pandas as pd
#0-49 Iris-setosa, 50-99 Iris-versicolor, 100-149 Iris-virginica
df = pd.read_csv('../input/iriscsv/Iris.csv')
df = df.drop(['Id'],axis=1)
vrsta = df['Species']
s = set()
for naziv in vrsta:
    s.add(naziv)
s = list(s) #kategorije Irisa = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
rows = list(range(50,100))
df = df.drop(df.index[rows]) #uzimamo prvih 50 za setosa i zadnjih 50 za virginica
print(df)
import matplotlib.pyplot as plt

iris_x = df['SepalLengthCm'] #Nazivi kolona u iris datasetu
iris_y = df['PetalLengthCm']
setosa_x = iris_x[:50]
setosa_y = iris_y[:50]
vriginica_x = iris_x[50:]
vriginica_y = iris_y[50:]

plt.figure(figsize=(8,6))
plt.scatter(setosa_x,setosa_y,marker='+',color='blue', label="Setosa")
plt.scatter(vriginica_x,vriginica_y,marker='o',color='red', label="Virginica")
plt.legend()
plt.show()
#Biblioteke koje se koriste
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches 
import random
import math

#Klasa dijeli dataset na 80% za treniranje i ostatak za testiranje
class MetodaPotpornihVektora(object):
    """
    Inicijalizacija, postavljanje boja i grafa
    """
    def __init__(self, dataA, dataB, crtanje=True,  prikaziSve=True):
        self.dataA = dataA
        self.dataB = dataB
        self.crtanje = crtanje
        self.prikaziSve = prikaziSve
        self.boje = {1:'r',-1:'b'}
        #Postavljanje grafa za 1 red, 1 kolonu (Ukoliko postoji više podgrafa)
        if self.crtanje:
            self.fig = plt.figure(figsize=(15,10))
            self.ax = self.fig.add_subplot(221)
            
    def postavi_podatke(self, dataA, dataB):
        """
        Postavljanje podataka u posebni niz koji drži tip i vrijednosti
        """
        for i in range(len(dataA)):
            dataA[i].insert(0, 1)
        for i in range(len(dataB)):
            dataB[i].insert(0, 0)

        data = dataA + dataB

        for i in range(len(data)):
            data[i].insert(0, 1)
            data[i] += [data[i].pop(1)]
            if data[i][3] == 1:
                data[i].append(-1)
            else:
                data[i].append(1)
            data[i].pop(3)
            data[i] = np.array(data[i])
            
        random.shuffle(data)
        return np.array(data)

    def odvoji_podatke(self, data_final):
        """
        Odvajanje Podataka
        """
        #Koristi se 80% za treniranje algoritma, dok je ostatak korišten za testiranje i gledanje preciznosti
        postotak80 = int((80/100 *len(data_final)))

        skup_testiranja = data_final[postotak80:]
        skup_treniranja = data_final[:postotak80]
        return skup_treniranja, skup_testiranja

    def train(self, Epoha = 100000):
        """
        Funkcija koja trenira algoritam kroz epohe. Generalno je postavljeno na 100000 epoha, dok zavisno kolika je veličina algoritama može se i smanjiti
        """
        #Funkcija preuzima broj epoha i niz podataka koje će koristiti za treniranje
        self.dataset = self.postavi_podatke(self.dataA, self.dataB)
        self.skup_treniranja, self.skup_testiranja = self.odvoji_podatke(self.dataset)
        
        skup_treniranja = self.skup_treniranja
        skup_testiranja = self.skup_testiranja
        
        w = np.array([0,0,0])
        stopa = 1
        konvergira = False
        k = 0
        min_distance_negative = 0 #nalaženje najmanje udaljenosti za marginu
        min_distance_positive = 0
        while(konvergira == False) and k < Epoha: #algoritam radi dok ne završi sa epohama ili dok ne nađe optimalnu hiperravan

            k +=1
            nasumican_vektor = int(random.uniform(0,len(skup_treniranja)-1))
            x_i = skup_treniranja[nasumican_vektor][:-1]
            y_i = skup_treniranja[nasumican_vektor][-1]

            n_t = 1/(k*stopa)

            if (y_i*np.dot(x_i,w)) < 1: # Ukoliko je y*(<w,x>) <1 moramo ažurirati vektor težine
                w[1:] = (1-n_t*stopa)*w[1:] + (n_t*y_i*x_i[1:])

                #Računanje biasa
                w[0] = w[0] + (y_i*x_i[0])
            else:
                w = (1-n_t*stopa)*w

            min_distance_positive, min_distance_negative, konvergira = self.provjeri_potporni(skup_treniranja, w) #Provjera na kojoj je strani hiperravni potporni vektor
        print("<------------------>\n")
        print("Broj iteracija do konvergiranja: ", k)
        
        self.prikazi(skup_treniranja, skup_testiranja,min_distance_negative, min_distance_positive, w,"Train Plot", "", "")
        
        return w, min_distance_negative, min_distance_positive

    def provjeri_potporni(self, skup_treniranja, w):
        """

        Here, we identify the support vectors for each weight vectort and see if it is equidistant from w

        """

        min_distance_positive = 999.0
        min_distance_negative = 999.0

        for i in skup_treniranja:
            x1 = i[1]
            x2 = i[2]
            y = i[3]
            try: #formula w/||w||
                d = abs(w[1]*x1 + w[2]*x2 + w[0])/ (math.sqrt(w[1]**2) + (math.sqrt(w[2]**2)))
                if y == -1:
                    if d<=min_distance_negative:
                        min_distance_negative = d
                else:
                    if d<=min_distance_positive:
                        min_distance_positive = d
            except: 
                pass

        if round(min_distance_positive,1) == round(min_distance_negative,1):
            return round(min_distance_positive)+0.6, round(min_distance_negative)+0.6, True
        else:
            return 1,1,False

    def prikazi(self, skup_treniranja, skup_testiranja, min_distance_negative, min_distance_positive, w, plot_type, naziv1, naziv2):
        """
        Jednostavna funkcija koja prelazi kroz podatke i iscrtava ih koristeći matplotlib biblioteku

        """

        if plot_type == "Train Plot":
            plt.scatter(skup_treniranja[:, 1], skup_treniranja[:, 2], c=skup_treniranja[:, 3],  edgecolor="black")
            ax = plt.gca()
            ax.set_facecolor('#ffeadb')
            ax.patch.set_alpha(0.5)
            xlim = ax.get_xlim()

            xx = np.linspace(xlim[0], xlim[1])
            yy = -(w[1]/w[2]) * xx - (w[0]/w[2])
            yy1 = min_distance_positive-(w[1]/w[2]) * xx - (w[0]/w[2])
            yy2 = -min_distance_negative-(w[1]/w[2]) * xx - (w[0]/w[2])
            plt.plot(xx, yy, c="r")
            plt.plot(xx, yy1, c="g", linestyle = "dashed")
            plt.plot(xx, yy2, c="g", linestyle = "dashed")
            plt.show()
        if plot_type == "Test Plot":
            plt.scatter(skup_treniranja[:, 1], skup_treniranja[:, 2], c=(skup_treniranja[:, 3]), edgecolor="black")
            plt.scatter(skup_testiranja[:, 1], skup_testiranja[:, 2], c='r', marker="D", edgecolor="black")
            ax = plt.gca()
            ax.set_facecolor('#ffeadb')
            ax.patch.set_alpha(0.5)
            xlim = ax.get_xlim()

            xx = np.linspace(xlim[0], xlim[1])
            yy = -(w[1]/w[2]) * xx - (w[0]/w[2])
            yy1 = min_distance_positive-(w[1]/w[2]) * xx - (w[0]/w[2])
            yy2 = -min_distance_negative-(w[1]/w[2]) * xx - (w[0]/w[2])
            plt.plot(xx, yy, c="r")
            plt.plot(xx, yy1, c="g", linestyle = "dashed")
            plt.plot(xx, yy2, c="g", linestyle = "dashed")
            
            predikcija_negativan = mpatches.Patch(color='darkblue', label=naziv1)
            predikcija_pozitivan = mpatches.Patch(color='y', label=naziv2)
            plt.legend(handles=[predikcija_negativan, predikcija_pozitivan])

            return plt


    def test(self, w, min_distance_negative, min_distance_positive, plot_type, naziv1, naziv2):
        """
        Funkcija za testiranje SVM Modela

        """
        plt = self.prikazi(self.skup_treniranja, self.skup_testiranja, min_distance_negative, min_distance_positive, w, plot_type, naziv1, naziv2)

        greske = 0
        for i in self.skup_testiranja:
            predikcija = np.dot(i[:-1], w)
            if predikcija<0 and i[-1] == 1:
                greske +=1
                plt.scatter(i[1], i[2], c='purple', edgecolor="white")
            elif predikcija>0 and i[-1] == -1:
                greske +=1
                plt.scatter(i[1], i[2], c='orange', edgecolor="white")
        print("\nUkupno tačaka trenirano: ", len(self.skup_treniranja))
        print("\nUkupno tačaka testirano: ", len(self.skup_testiranja))
        print("\nBroj pogrešnih tačaka: ", greske)
        print("\nPreciznost: ", ((len(self.skup_testiranja) - greske) / len(self.skup_testiranja)) * 100, "%")
        print("\n<------------------>")

        plt.show()
%matplotlib inline
import numpy as np

setosa_data = [[a,b] for a, b in zip(setosa_x, setosa_y)]
virginica_data = [[a,b] for a, b in zip(vriginica_x, vriginica_y)] 
#Postavljanje podataka, -1 - setosa
#                        1 - virginica


#Generisanje skupa podataka za testiranje i obučavanje
svm = MetodaPotpornihVektora(setosa_data, virginica_data)
#Nalaženje vektora težine i potpornih vektora na 150000 iteracija
w, min_distance_negative, min_distance_positive = svm.train()
#Testiranje SVM Modela
svm.test(w, min_distance_negative, min_distance_positive, "Test Plot", "Setosa", "Virginica")
import numpy as np

def funkcija_troska(h, y):
    return np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

funkcija_troska(1/2, 1)
#Sigmoidna funkcija (postavlja vrijednosti između 0 i 1)
import math
def sigmoid_funkcija(vrijednost):
    rezultat = 1/(1+math.e**(-vrijednost))
    return rezultat

print(sigmoid_funkcija(10), sigmoid_funkcija(1))
import numpy as np
import time

#Postavljamo brzinu obučavanja kao i broj iteracija.

class LogistickaRegresija:
    def __init__(self, X, brzina_obucavanja=0.1, broj_iteracija=15000):
        self.brzina_u = brzina_obucavanja
        self.broj_it = broj_iteracija

        # m je broj podataka, dok je n broj osobina
        self.m, self.n = X.shape

    def train(self, X, y):
        # inicijalizacija tezine i biasa/slobodnog člana
        self.tezinski_vektor = [[0. for kolone in range(1)] for red in range(self.n)]
        self.bias = 0

        #treniranje kroz petlju za num_iteracija
        for it in range(self.broj_it + 1):
            #računanje hipoteze
            y_predict = sigmoid_funkcija(np.dot(X, self.tezinski_vektor) + self.bias)

            # cilj umanjivanja troska/cijene
            cijena = -1 / self.m * funkcija_troska(y_predict, y) #optimizacija

            # racunanje gradijent spusta (glavni cilj je minimiziranjem troska)
            #skalarni proizvod transponovane matrice X sa y_predict - y
            dw = 1 / self.m * np.dot(X.T, (y_predict - y))
            db = 1 / self.m * np.sum(y_predict - y)

            # gradient descent azuriraj korak
            self.tezinski_vektor -= self.brzina_u * dw
            self.bias -= self.brzina_u * db

            # nakon 2000-te iteracije ispisati trosak
            if it % 2000 == 0:
                print(f"Trošak iteracije {it} -> {cijena}")
                
        self.y_predict = y_predict

        return self.tezinski_vektor, self.bias

    def predikcija(self, X):
        y_predict = sigmoid_funkcija(np.dot(X, self.tezinski_vektor) + self.bias)
        y_predict_rezultat = y_predict > 0.5 #provjera da li je veće od 0.5, u tom slučaju pripada klasi 1

        return y_predict_rezultat
    
    def prikaz_podataka(self, dataset1, dataset2, X, y, y_predict, naziv1, naziv2):
        trening_set_velicina = len(y)-len(y_predict)
        ax = plt.gca()
        #Prikaz rezultata na grafu
        x_vrijednost = [np.array(dataset1).min(),np.array(dataset2).max()]
        y_vrijednost = -(x_vrijednost * w[0] + b) / w[1] 
        
        
        ax.title.set_text("Prikaz granice odluke logističke regresije i predikcija")
        ax.set_facecolor('#ffeadb')
        ax.patch.set_alpha(0.5)
        xlim = ax.get_xlim()

        xx = np.linspace(xlim[0], xlim[1])
        
        plt.scatter(dataset1[0],dataset1[1],marker='D',color='blue', edgecolor="black", label=naziv1)
        plt.scatter(dataset2[0],dataset2[1],marker='o',color='red', edgecolor="black", label=naziv2)

        for i in range(len(y_predict)):
            if (y[trening_set_velicina + i]!=y_predict[i]):
                if (y[i] == 0):
                    ax.scatter(X[trening_set_velicina + i][0],X[trening_set_velicina+i][1],marker='x',color='orange')
                else:
                    ax.scatter(X[trening_set_velicina+i][0],X[trening_set_velicina+i][1],marker='x',color='green')
            else:
                ax.scatter(X[trening_set_velicina+i][0],X[trening_set_velicina+i][1],marker='d',color='white', edgecolor="black")  
        ax.legend()  
        ax.plot(x_vrijednost,y_vrijednost, c="r")
        
#Postavljanje podataka iz Iris dataset-a i pokretanje logističke regresije (treniranje 80%, testiranje 20%)
prvaKlasa = list(map(list, zip(setosa_x, setosa_y)))
drugaKlasa = list(map(list, zip(vriginica_x, vriginica_y)))

datasetIris = np.array(prvaKlasa + drugaKlasa) #podaci originalni
klasifikacijaF1 = [[0]] * len(prvaKlasa)
klasifikacijaF2 = [[1]] * len(drugaKlasa)
klasifikacijaTestiranje = klasifikacijaF1 + klasifikacijaF2 #njihove klase

#Nasumično vraćanje podataka
c = list(zip(datasetIris, klasifikacijaTestiranje))

random.shuffle(c)

X, y = zip(*c)
X = np.array(X)
y = np.array(y)

print("Skup podataka dužine: ", len(X))

#Pokretanje algoritma logističke regresije 
broj_treniranja = 80 #80% za treniranje, ostatak testiranje
logreg = LogistickaRegresija(X) #inicijalizacija (uzima samo veličinu podataka)
start = time.process_time()
w, b = logreg.train(X[:broj_treniranja], y[:broj_treniranja]) #treniranje
print(f'Vrijeme treniranja SVM-a: {time.process_time() - start}s')
y_predict = logreg.predikcija(X[broj_treniranja:]) #predikcija
logreg.prikaz_podataka([setosa_x, setosa_y], [vriginica_x, vriginica_y], X, y, y_predict, "Setosa", "Virginica") #prikaz podataka
print(f"\nPreciznost: {np.sum(y[broj_treniranja:]==y_predict)/len(y[broj_treniranja:]) * 100} %") #preciznost
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
import time

iris = datasets.load_iris()
X = iris.data[:100, :2]
y = iris.target[:100]

def sklearnBibliotekeSVM(X, y, ravel=False):
    h = .02  # step size in the mesh
    x_train,x_test,y_train,y_test=train_test_split(X,y,train_size=0.8,test_size=0.2)
    C = 100.0  # SVM regularization parameter
    svc = svm.SVC(kernel='linear', C=C).fit(X, y)
    start = time.process_time()
    if ravel:
        rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(x_train, y_train.values.ravel())
        print(f'Vrijeme treniranja RBF Kernel SVM-a: {time.process_time() - start}s')
        start = time.process_time()
        poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(x_train, y_train.values.ravel())
        print(f'Vrijeme treniranja Polinomijalnog Kernel SVM-a: {time.process_time() - start}s')
    else:
        rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(x_train, y_train)
        print(f'Vrijeme treniranja RBF Kernel SVM-a: {time.process_time() - start}s')
        start = time.process_time()
        poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(x_train, y_train)
        print(f'Vrijeme treniranja Polinomijalnog Kernel SVM-a: {time.process_time() - start}s')

    titles=["Metoda potpornih vektora sa RBF Kernel funkcijom", "Metoda potpornih vektora sa Polinomijalnom Kernel funkcijom"]

    # mesh plot
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    start = time.process_time()

    naziv= ["RBF", "Polinomijalnog"]
    print('\n')
    for i, clf in enumerate((rbf_svc, poly_svc)):  
        plt.subplot(1, 1 + i, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        start = time.process_time()
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        print(f'Preciznost: {clf.score(x_test, y_test)}')
        print(f'Vrijeme predikcije {naziv[i]} Kernel SVM-a: {time.process_time() - start}s')
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

        # plot treniranja
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(titles[i])

        plt.show()
        
sklearnBibliotekeSVM(X, y)    
#Support Vector Machine
import random
import matplotlib.patches as mpatches

#CPU vrijeme
import time


#Prikaz SVM i Logističke regresije
def treniranjeTestiranjeSVMLR(datasetF1, datasetF2, naziv1, naziv2, xlimitacija, ylimitacija, crtat=True):
    #----------LR-----------
    #postavljanje podataka na 80% i 20% treniranje i testiranje
    datasetLR1 = datasetF1
    datasetLR2 = datasetF2
    datasetMain = datasetLR1 + datasetLR2
    klasifikacijaF1 = [[0]] * len(datasetLR1)
    klasifikacijaF2 = [[1]] * len(datasetLR2)
    klasifikacijaMain = klasifikacijaF1 + klasifikacijaF2

    c = list(zip(datasetMain, klasifikacijaMain))
    
    random.shuffle(c)

    X, y = zip(*c)
    X = np.array(X)
    y = np.array(y) 

    broj_treniranja = int((80 / 100) * (len(X)))
    
    #Pokretanje logističke regresije
    start2 = time.process_time()
    logreg = LogistickaRegresija(X)
    print(f'Vrijeme treniranja logističke regresije: {time.process_time() - start2}s')
    w, b = logreg.train(X[:broj_treniranja], y[:broj_treniranja])
    start2test = time.process_time()
    
    y_predict = logreg.predikcija(X[broj_treniranja:])
    print(f'Vrijeme testiranja logističke regresije: {time.process_time() - start2test}s')
    print(f"\nPreciznost: {np.sum(y[broj_treniranja:]==y_predict)/len(y[broj_treniranja:]) * 100} %") #preciznost
    
    #Prikaz rezultata na grafu
    
    #Gledamo prikaz 50 podataka na grafu
    podatak_jedan_X = []
    podatak_jedan_Y = []
    podatak_nula_X = []
    podatak_nula_Y = []
    for index, klasa in enumerate(y):
        if (klasa == 0):
            podatak_nula_X.append(X[index][0])
            podatak_nula_Y.append(X[index][1])
        else:
             podatak_jedan_X.append(X[index][0])
             podatak_jedan_Y.append(X[index][1])
    if crtat:     
        trening_set_velicina = len(y)-len(y_predict)
        ax = plt.gca()
        #Prikaz rezultata na grafu
        
        
        a = -w[0] / w[1]
        xx = np.linspace(X[:][0].min(), X[:][0].max())
        yy = a * xx - b / w[1]

        ax.plot(xx, yy, 'k-', c="r")
        #x_vrijednost = [np.array(dataset1).min(),np.array(dataset2).max()]
        #y_vrijednost = -(x_vrijednost * w[0] + b) / w[1] 
        #ax.plot(x_vrijednost,y_vrijednost, c="r")  
        
        ax.title.set_text("Prikaz granice odluke logističke regresije i predikcija")
        ax.set_facecolor('#ffeadb')
        ax.patch.set_alpha(0.5)
        xlim = ax.get_xlim()

        xx = np.linspace(xlim[0], xlim[1])
        ax.set(aspect="equal",
           xlim=(xlimitacija[0], xlimitacija[1]), ylim=(ylimitacija[0], ylimitacija[1]),
           xlabel="$X_1$", ylabel="$X_2$")
        ax.scatter(podatak_nula_X,podatak_nula_Y,marker='D',color='blue', edgecolor="black", label=naziv1)
        ax.scatter(podatak_jedan_X,podatak_jedan_Y,marker='o',color='red', edgecolor="black", label=naziv2)

        for i in range(len(y_predict)):
            if (y[trening_set_velicina + i]!=y_predict[i]):
                if (y[i] == 0):
                    ax.scatter(X[trening_set_velicina + i][0],X[trening_set_velicina+i][1],marker='x',color='orange')
                else:
                    ax.scatter(X[trening_set_velicina+i][0],X[trening_set_velicina+i][1],marker='x',color='green')
            else:
                ax.scatter(X[trening_set_velicina+i][0],X[trening_set_velicina+i][1],marker='d',color='white', edgecolor="black")  
        ax.legend()  
    time.sleep(3) 
    #----------SVM-----------
    #treniranje je 80% dataset-a

    #Generisanje skupa podataka za testiranje i obučavanje
    dataset1 = datasetF1
    dataset2 = datasetF2
    svm = MetodaPotpornihVektora(dataset1, dataset2)
    #Nalaženje vektora težine i potpornih vektora na 150000 iteracija
    start1 = time.process_time()
    w, min_distance_negative, min_distance_positive = svm.train()
    print(f'Vrijeme treniranja metode potpornih vektora: {time.process_time() - start1}s')
    #Testiranje SVM Modela
    start1test = time.process_time()
    svm.test(w, min_distance_negative, min_distance_positive, "Test Plot", naziv1, naziv2)
    print(f'Vrijeme testiranja metode potpornih vektora: {time.process_time() - start1test}s\n\n\n')

#Klasifikacija - gaussian isotrpic blobs
from sklearn.datasets import make_blobs

np.random.seed(1)

center_box = (0, 50) # okolika u kojoj ce se nalaziti
standard_dev = 3 # standardna devijacija podataka (veće odstupanje)

X, y = make_blobs(n_samples=1000, centers=2, center_box=center_box, cluster_std=standard_dev)
y = y[:, np.newaxis]

print("Prvih 10 podataka dataset-a:\n", X[:10], "\nKlasa:\n", list(y[:10]))

blob_klasa_jedan = []
blob_klasa_nula = []
for index, klasa in enumerate(y):
    if (klasa == 0):
        blob_klasa_nula.append(X[index])
    else:
         blob_klasa_jedan.append(X[index])
            
plt.scatter([item[0] for item in blob_klasa_nula],[item[1] for item in blob_klasa_nula],marker='+',color='red', label='Klasa 1')
plt.scatter([item[0] for item in blob_klasa_jedan],[item[1] for item in blob_klasa_jedan],marker='o',color='blue', label="Klasa 2")
plt.legend()
#Support Vector Machine
import random

#CPU vrijeme
import time
#Obučavanje skupa podataka na obje metode i pregled preciznosti
for index, i in enumerate(blob_klasa_nula):
    blob_klasa_nula[index] = [blob_klasa_nula[index][0], blob_klasa_nula[index][1]]
for index, i in enumerate(blob_klasa_jedan):
    blob_klasa_jedan[index] = [blob_klasa_jedan[index][0], blob_klasa_jedan[index][1]]

treniranjeTestiranjeSVMLR(blob_klasa_nula, blob_klasa_jedan, "Blob Set 1", "Blob Set 2", [-10,50], [-10,50])
#Logistička regresija
import matplotlib.patches as mpatches

logreg = LogistickaRegresija(X)
w, b = logreg.train(X[:10], y[:10])
start = time.process_time()
y_predict = logreg.predikcija(X)
print(f'Vrijeme treniranja logističke regresije: {time.process_time() - start}s')
#Prikaz rezultata na grafu
ax = plt.gca()

#Gledamo prikaz 50 podataka na grafu, dok testiramo nad ostalim
blob_klasa_LG_jedan_X = []
blob_klasa_LG_jedan_Y = []
blob_klasa_LG_nula_X = []
blob_klasa_LG_nula_Y = []
for index, klasa in enumerate(y):
    if (klasa == 0):
        blob_klasa_LG_nula_X.append(X[index][0])
        blob_klasa_LG_nula_Y.append(X[index][1])
    else:
         blob_klasa_LG_jedan_X.append(X[index][0])
         blob_klasa_LG_jedan_Y.append(X[index][1])
        
x_vrijednost = [X.min(),X.max()]
y_vrijednost = -(x_vrijednost * w[0] + b) / w[1] 

plt.scatter(blob_klasa_LG_nula_X,blob_klasa_LG_nula_Y,marker='+',color='blue')
plt.scatter(blob_klasa_LG_jedan_X, blob_klasa_LG_jedan_Y,marker='o',color='red')
plt.plot(x_vrijednost,y_vrijednost)

#Broj tačnih podataka kroz 1000
print(f"Preciznost: {np.sum(y==y_predict)/X.shape[0] * 100} %")
for i in range(len(y_predict)):
    if (y[i]!=y_predict[i]):
        if (y[i] == 0):
            plt.scatter(X[i][0],X[i][1],marker='x',color='orange')
        else:
            plt.scatter(X[i][0],X[i][1],marker='x',color='green')
            
pogresna_predikcija_negativan = mpatches.Patch(color='orange', label='False Negative')
pogresna_predikcija_pozitivan = mpatches.Patch(color='green', label='False Positive')
plt.legend(handles=[pogresna_predikcija_negativan, pogresna_predikcija_pozitivan])
X, y = make_blobs(n_samples=1000, centers=2, center_box=center_box, cluster_std=standard_dev)

sklearnBibliotekeSVM(X, y)
df = pd.read_csv('../input/weight-height/weight-height.csv')
df = df.sample(frac = 1) 
print("Broj podataka:", len(df), "\n", df[:5])


maleTezina = []
maleVisina = []
femaleTezina = []
femaleVisina = []

datasetMusko = []
datasetZensko = []
for index in range(len(df)):
    data = df.iloc[index]
    if data[0] == 'Male':
        maleVisina.append(data[1])
        maleTezina.append(data[2])
        datasetMusko.append([data[1], data[2]])
    else:
        femaleVisina.append(data[1])
        femaleTezina.append(data[2])
        datasetZensko.append([data[1], data[2]])

plt.scatter(maleVisina,maleTezina,marker='+',color='blue', label="Muško")
plt.scatter(femaleVisina,femaleTezina,marker='o',color='red', label='Žensko')
plt.legend()
plt.show()

treniranjeTestiranjeSVMLR(datasetMusko, datasetZensko, "Muška Osoba", "Ženska Osoba", [50,250], [50, 250])
df = pd.read_csv('../input/weight-height/weight-height.csv')
df = df.sample(frac = 1) 
Xset3 = []
yset3 = []

for index in range(len(df)):
    data = df.iloc[index]
    if data[0] == 'Male':
        yset3.append(0)
    else:
        yset3.append(1)
    Xset3.append([data[1], data[2]])
X = np.array(Xset3)
y = np.array(yset3)

#sklearnBibliotekeSVM(Xset3, yset3)


h = .02  # step size in the mesh

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C).fit(X, y)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)


# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# title for the plots
titles=["SVM sa RBF Kernel f", "SVM sa Polinomijalnom Kernel f"]

for i, clf in enumerate((rbf_svc, poly_svc)):
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()
from sklearn import preprocessing

df = pd.read_csv('../input/room-occupancy/file.csv')[['Light', 'Temperature', 'Occupancy']]
df = df.sample(frac = 1) 
df = df[:100]
#df = (df-df.min())/(df.max()-df.min())

print("Broj podataka:", len(df), "\n", df[:5])


zauzetoLight = []
zauzetoTemperatura = []
nijeZauzetoLight = []
nijeZauzetoTemperatura = []

datasetZauzeto = []
datasetNijeZauzeto = []

for index in range(len(df)):
    data = df.iloc[index]
    if data[2] == True:
        zauzetoLight.append(data[0])
        zauzetoTemperatura.append(data[1])
        datasetZauzeto.append([data[0], data[1]])

    elif data[2] == False:
        nijeZauzetoLight.append(data[0])
        nijeZauzetoTemperatura.append(data[1])
        datasetNijeZauzeto.append([data[0], data[1]])

plt.scatter(zauzetoLight,zauzetoTemperatura,marker='+',color='red', label="Zauzeto")
plt.scatter(nijeZauzetoLight,nijeZauzetoTemperatura,marker='o',color='blue', label='Nije Zauzeto')
plt.legend()
plt.show()

treniranjeTestiranjeSVMLR(datasetZauzeto, datasetNijeZauzeto, "Zauzeta Soba", "Slobodna soba", [0,1000], [0, 25])
df = pd.read_csv('../input/room-occupancy/file.csv')[['Light', 'Temperature', 'Occupancy']]
df = df.sample(frac = 1) 

Xset4 = []
yset4 = []

for index in range(len(df)):
    data = df.iloc[index]
    if data[2] == True:
        yset4.append(0)
    elif data[2] == False:
        yset4.append(1)
    Xset4.append([data[0], data[1]])

Xset4 = np.array(Xset4)
yset4 = np.array(yset4)

sklearnBibliotekeSVM(Xset4, yset4)
from sklearn import preprocessing

df = pd.read_csv('../input/swiss-banknote-conterfeit-detection/banknotes.csv')[['Bottom', 'Top', 'conterfeit']]
df = df.sample(frac = 1) 
#df = (df-df.min())/(df.max()-df.min())

print("Broj podataka:", len(df), "\n", df[:5])


originalneDole = []
originalneGore = []
falsifikovaneDole = []
falsikifovaneGore = []

datasetORIGINAL = []
datasetFALSIFIKAT = []
for index in range(len(df)):
    data = df.iloc[index]
    if data[2] == True:
        originalneDole.append(data[0])
        originalneGore.append(data[1])
        datasetORIGINAL.append([data[0], data[1]])
    elif data[2] == False:
        falsifikovaneDole.append(data[0])
        falsikifovaneGore.append(data[1])
        datasetFALSIFIKAT.append([data[0], data[1]])
print(len(datasetZensko), len(datasetMusko))
plt.scatter(originalneDole,originalneGore,marker='+',color='blue', label="Originalne")
plt.scatter(falsifikovaneDole,falsikifovaneGore,marker='o',color='red', label='Falsifikovane')
plt.legend()
plt.show()

treniranjeTestiranjeSVMLR(datasetORIGINAL, datasetFALSIFIKAT, "Zauzeta Soba", "Slobodna soba", [0,15], [0, 15])
df = pd.read_csv('../input/swiss-banknote-conterfeit-detection/banknotes.csv')[['Bottom', 'Top', 'conterfeit']]
df = df.sample(frac = 1) 

Xset5 = []
yset5 = []

for index in range(len(df)):
    data = df.iloc[index]
    if data[2] == True:
        yset5.append(-1)
    elif data[2] == False:
        yset5.append(0)
    Xset5.append([data[0], data[1]])

X = np.array(Xset5)
y = np.array(yset5)

#sklearnBibliotekeSVM2(Xset5, yset5)

h = .02  # step size in the mesh

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C).fit(X, y)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)


# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# title for the plots
titles=["SVM sa RBF Kernel f", "SVM sa Polinomijalnom Kernel f"]

for i, clf in enumerate((rbf_svc, poly_svc)):
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()