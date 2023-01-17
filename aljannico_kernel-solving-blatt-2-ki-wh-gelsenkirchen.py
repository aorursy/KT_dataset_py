from time import time



# Some magic to time the calls to functions. timing is a decorator, usage see below:

def timing(f):

    def wrap(*args):

        time1 = time()

        ret = f(*args)

        time2 = time()

        print('%s function took %0.3f ms' % (f.__name__, (time2-time1)*1000.0))

        return ret

    return wrap



def seq(n,k):

    if n==0:

        return 0

    if k==1:

        return n

    return seq(n-1,k) + seq(n-1,k-1)



@timing

def seq_top_call(n,k):

    return seq(n,k)



@timing

def iter(n,k):

    result = 1 # nicht komplett äquivalent zu seq, wir behandeln die Grenzfälle hier nicht

    for i in range(1,k+1):  # i aus [1,k]

        result *= ((n-k+i) / i) # // does not work here!

    return result



test = [(4,2),(5,3),(10,2),(30,8),(100,2)]

for n,k in test:

    print("Calling iter(",n,",",k,"): ",iter(n,k))

    print("Calling seq(",n,",",k,"): ",seq_top_call(n,k))

    

# seq is MUCH slower than iter, costs exploding exponentially...



print("Calling iter(",500,",",10,"): ",iter(500,10)) 

# print("Calling iter(",10000,",",500,"): ",iter(10000,500)) # Too large if computed with floats..infinity

# Watch it: seq(500,10) and seq(10000,500) will not work, don't try it at kaggle - only at home ;)
# from matplotlib import pyplot

from pylab import *

import numpy as np



# Our data

data = [(21,3), (22,8), (23,7), (24,3), (25,4), (26,1), (27,1), (28,0), (29,1)]

raw_data = [ 21,21,21,22,22,22,22,22,22,22,22,23,23,23,23,23,23,23,

    24,24,24,25,25,25,25,26,27,29 ]



n = len(raw_data)



# A4.1: Stab-, Balken- und Kreisdiagramm zu den realativen Häufigkeiten



labels,sizes = zip(*data)

print(labels,sizes)



# Bar chart

fig1, ax1 = plt.subplots()

ax1.bar(labels, sizes)

ax1.set_title('Säulen- bzw. Stabdiagramms')



# Pie chart

fig2, ax2 = plt.subplots()

ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=0)

ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax2.set_title('Kreisdiagramms')





# Show the pie!

plt.show()



# Balkendiagramm (horizontal)

fig3,ax3 = plt.subplots()

ax3.hlines(labels, [0], sizes , lw=7)

ax3.set_title('Balkendiagramm')



plt.show()
# Es folgt Aufgabe 4.2, ein Histogramm zu den angegbenen Intervallen:



hist(raw_data,bins=[19.999,21.999,23.999,25.999,27.999,29.999],normed=True,

    cumulative=False) 

title("Histogramm")

show()



# ... und die empirische Verteilungsfunktion (A4.3)

hist(raw_data,bins=[20,21,22,23,24,25,26,27,28,29,30],normed=True,

    cumulative=True) #,histtype="step") 



title("Empirische Verteilungsfunktion F")

show()
# A4.4: Bestimmen Sie das arithmetische Mittel, den Median und den Modus der Daten

# (wir könnten auch Funktionen z.B. aus den numpy nutzen, wir machen es hier ab "zu Fuss")

## Mittelwert



def mittelwert(data,dim=1):

    '''

    This looks a little more complex as it can give you average value

    in multiple dimensions in one go.

    Example: data = [ (1,2,3),(2,3,4) ] will give you [1.5,2.5,3.5]

    Example: data = [1,2,3] will give you 2

    '''

    res = []

    n = len(data)

    for i in range(dim):

        res.append(0.0)

    for d in data:

        for i in range(dim):

            if dim == 1:

                res[i] += d

            else:

                res[i] += d[i]

    for i in range(dim):

        res[i] /= n

    return res[0] if dim==1 else res

    

## Median

def median(data):

    '''

    For an odd amount of data, take the value in the middle position.

    Otherwise, average over the two elements next to the middle position.

    Example: len(data) = 11, 0..10, mid position: (11-1) / 2 = 10 / 2 = 5 =>

      [0..4,5,6..10]

    Example: len(data) = 10, 0..9, mid positions: (n-2)/2 = 4,(n-2)/2+1 = 5

    '''

    n = len(data)

    if n % 2: # odd

        return data[(n-1)//2] # not like on the slides, as we count from 0!

    else:

        return (data[(n-2)//2] + data[(n-2)//2+1]) / 2

    

## Modus

def modus(data):

    maxpos,maxval = data[0]

    for d in data:

        if d[1] > maxval:

            maxpos,maxval = d

    return maxpos



## Teilaufgabe 4

mu = mittelwert(raw_data)

print("Mittelwert: ",mu)

med = median(raw_data)

print("Median: ",med)

mod = modus(data)

print("Modus: ",mod)
# A4.5: Verwenden Sie die Beziehung zwischen arithmetischem Mittel, Median und Modus, um eine Aussage zu

# Steilheit bzw. Symmetrie der Verteilung treffen zu können.



## Schiefe, Folie 18

def schiefe(mu,med,modus):

    if mu > med and med > mod:

        print("Schiefe der Verteilung: linkssteil bzw. rechtsschief")

        return -1

    if mu < med and med < mod:

        print("Schiefe der Verteilung: rechtssteil bzw. linkssschief")

        return 1

    print("Symmetrisch oder annähernd symmetrisch")

    return 0



schiefe_result = schiefe(mu,med,mod)
# A4.6: Bestimmen Sie die Empirische Varianz, die Standardabweichung und die Stichprobenvarianz.



## (Empirische) Varianz (eindimensional)

def varianz(data,mu=None,stichprobe=False):

    if mu == None:

        mu = mittelwert(data)

    n = len(data)

    sum = 0

    for d in data:

        sum += (d - mu)**2

    return sum / (n-1) if stichprobe else sum / n

    

## Standardabweichung

def standardabweichung(data,mu=None,stichprobe=False):

    return sqrt(varianz(data,mu,stichprobe))



var   = varianz(raw_data)

print("Varianz: ",var)

sigma = sqrt(var)

print("Standardabweichung: ",sigma)

svar   = varianz(raw_data,stichprobe=True)

print("Stichprobenvarianz: ",svar)

ssigma = standardabweichung(raw_data,stichprobe=True)

print("Standardabweichung zur Stichprobenvarianz: ",ssigma)
# A4.7_ Transformieren Sie die Daten von Menschenjahren in Hundejahre (1 Menschenjahr = 7 Hundejahre). 

# Bestimmen Sie auf einfache Art (empirische) Varianz und Standardabweichung der entstehenden Daten.



# Teilaufgabe 8

print("Hundejahre-Varianz =",var * (7**2))

print("Hundjahres-Standardabweichung =", abs(7)*sigma)
# A4.8:  Zerlegen Sie die Daten in die 5 Intervalle, die sie bereits beim Histogramm verwendet haben. 

# Diese Intervalle sind nun unsere Schichten. Bestimmen Sie nun die Streuung in den Schichten und

# die Streuung zwischen den Schichten und, abschließend, die Gesamtstreuung (s. Folien 23 und 24).



def split(data,splitpoints):

    cdata = data[:]

    res = []

    for i in range(len(splitpoints)):

        res.append([])

    for pos,split in enumerate(splitpoints):

        check = cdata[:]

        cdata = []

        for d in check:

            if d < split:

                res[pos].append(d)

            else:

                cdata.append(d)

    return res



def streuungs_mittelwert(klasses):

    n = 0

    sum = 0

    for klass in klasses:

        nj = len(klass)

        xj = mittelwert(klass)

        sum += nj * xj

        n += nj

    if n == 0: return 0

    return sum / n



def inner_class_streuung(klasses):

    sum = 0

    n = 0

    for klass in klasses:

        nj = len(klass)

        sum += nj * varianz(klass)

        n += nj

    if n == 0: return 0

    return sum / n    



def intra_class_streuung(klasses):

    mu = streuungs_mittelwert(klasses)

    sum = 0

    n = 0

    for klass in klasses:

        nj = len(klass)

        xj = mittelwert(klass)

        sum += nj * (xj - mu)**2

        n += nj

    if n == 0: return 0

    return sum / n    

        

# Teilaufgabe 9

splitted_data = split(raw_data,[22,24,26,28,30])

print(splitted_data)

print("Streuungsmittelwert: ",streuungs_mittelwert(splitted_data))

in_class_var = inner_class_streuung(splitted_data)

print("Inner-Class-Streuung: ", in_class_var)

int_class_var = intra_class_streuung(splitted_data)

print("Intra-Class-Streuung: ", int_class_var)

print("Varianz aus der Streuungszerlegung: ", in_class_var + int_class_var)
# A4.9: Bestimmen sie den Momentkoeffizienten der Schiefe für unsere Daten und treffen Sie eine Aussage zur

# Schiefe bzw. Symmetrie der Daten. Vergleichen Sie ihr Resultat mit dem oben erzielten



## Momentkoeffizient der Schiefe

def moment_schiefe(data,mu=None):

    if mu == None:

        mu = mittelwert(data)

    n = len(data)

    sum = 0

    for d in data:

        sum += (d - mu)**3

    m3 = sum / n

    gm = m3 / (standardabweichung(data,mu)**3)

    if gm > 0:

        print("Verteilung ist linkssteil")

    elif gm < 0:

        print("Verteilung ist rechtssteil")

    else:

        print("Verteilung ist symmetrisch")

    return gm    



print("Momentkoeffizient der Schiefe: ",moment_schiefe(raw_data))
# A4.10: Bestimmen sie das Wölbungmaß nach Fisher. 

# Ist die Verteilung spitzer oder flacher als die Normalverteilung?



def moment_fisher(data,mu=None):

    if mu == None:

        mu = mittelwert(data)

    n = len(data)

    sum = 0

    for d in data:

        sum += (d - mu)**4

    m4 = sum / n

    gamma = m4 / (standardabweichung(data,mu)**4) - 3

    if gamma > 0:

        print("Verteilung ist spitzer, als die Normalverteilung")

    elif gamma < 0:

        print("Verteilung ist flacher, als die Normalverteilung")

    else:

        print("Verteilung woelbt sich wie die Normalverteilung")

    return gamma   



print("Woelbung nach Fisher: ",moment_fisher(raw_data))



# A4.11: Ist die Verteilung uni-, bi- oder multimodal?

print("Die Verteilung ist multimodal (3 lokale Maxima).")
# Versuchen Sie, Parameter µ und σ für eine Normalverteilung so zu wählen, dass sie sich dem Verlauf

# des Histogramms möglichst gut annähert. Hier ist mit “möglichst gut” gemeint, dass sie dies nicht

# schlechter tun soll, als eine beliebige andere Normalverteilung mit anders belegten Parametern.



# Teilaufgabe 2 und 13

a,b,c = hist(raw_data,bins=[19.999,21.999,23.999,25.999,27.999,29.999],normed=True)

title("Histogramm der Altersdaten mit Normalverteilungsschaetzer")

xs = np.arange(20.0,30.0,0.01) # Viele Stützpunkte zwischen [20 und 30) im Abstand von 0.01



plot(xs,normpdf(xs,mu,ssigma)) # using deviation from stichprobenvarianz

plot(xs,normpdf(xs,mu,sigma)) # Using standard deviation

plot(xs,normpdf(xs,mu,1.4)) # steeper

show()



# Eine typische Wahl sind die Standardabweichungen von empirischer bzw. Stichprobenvarianzen

# "Besser" geht es ohne weiteres Wissen nicht. Hier mag das - gegen die in Intervalle zusammengefassten

# Daten aufgetragen - nicht so richtig perfekt aussehen.
# A4.13: Zeichnen Sie einen NQ-Plot zu unseren Daten.



def gaussfunc(y,ybar,sigma):

    """

    cumulative normal distribution function of the variable y

    with mean ybar,standard deviation sigma

    uses expression 7.1.26 from Abramowitz & Stegun

    accuracy better than 1.5e-7 absolute

    """

    x=(y-ybar)/(math.sqrt(2.)*sigma)

    t=1.0/(1.0 + .3275911*abs(x))

    erf=1.0 - math.exp(-x*x)*t*(.254829592 -t*(.284496736-t*(1.421413741-t*(1.453152027 -t*1.061405429))))

    erf=abs(erf)

    sign=x/abs(x)

    return 0.5*(1.0+sign*erf)



def gauss_quantil(p):

    xx = arange(-3.0,3.0,0.01)

    for x in xx:

        v = gaussfunc(x,0,1)

        if v > p:

            return x-0.005



def nq_data(raw_data):

    res = []

    n = len(raw_data)

    for i in range(1,n+1):

        p = (i-0.5)/float(n)

        res.append((gauss_quantil(p),raw_data[i-1]))

    return res



# Show some stuff

def show_result(data,titel=""):

    x = [i[0] for i in data]

    y = [i[1] for i in data]

    scatter(x,y,s=2, marker='^', c='b')

    title(titel)

    show()

    

## Teilaufgabe 14

show_result(nq_data(raw_data),"NQ-Plot")
def transform(data,scale = 1.0):

    '''

    Erzeugen der Daten, wie in Aufgabe 6.15 gewünscht.

    '''

    res = []

    for d in data:

        j = d[0]

        anzahl = d[1]

        if anzahl:

            intervall = float(scale / anzahl)

            for i in range(anzahl):

                position = j + intervall/2 + intervall*i

                res.append(position)

    return res



    

## Kerne

def rechteck_dichte(u):

    if u >= -1 and u <= 1:

        return 0.5

    return 0



def epanechnikov_dichte(u):

    if u >= -1 and u <= 1: 

        return 0.75*(1-u*u)

    return 0



def schaetzer(x,h,dichte):

    sum = 0

    n = len(realdata)

    for xi in realdata:

        sum += dichte((x-xi) / h)

    return sum / n*h



def verlauf(h,dichte,links=20.0,rechts=30.0,step=0.05):

    res = []

    position = links

    while position < rechts:

        wert = schaetzer(position,h,dichte)

        res.append((position,wert))    

        position += step    

    return res

print(transform(data))



def plot_data(h,dichte,dichte_name):

    titel = dichte_name + " mit Fenster %1.1f" % h

    if __debug__: print("Gefundene Schaetzungen für",titel)

    plotdata = verlauf(h,dichte)

    if __debug__: print("   ",plotdata)

    show_result(plotdata,titel)

    





## Teilaufgabe 15

realdata = transform(data)    

print("Erzeugte Ausprägungen:")

print("   ",realdata)

bs = np.arange(20.0,29.5,0.5)

hist(realdata,bins=bs,normed=True)

title("Histogramm der erzeugten Auspraegungen")

show()



show_result(nq_data(realdata),"NQ-Plot fuer die gestreckten Daten")



plot_data(0.2,rechteck_dichte,"Rechteck")

plot_data(0.4,rechteck_dichte,"Rechteck")

plot_data(0.8,rechteck_dichte,"Rechteck")

plot_data(1.2,rechteck_dichte,"Rechteck")



plot_data(0.2,epanechnikov_dichte,"Epanechnikov")

plot_data(0.4,epanechnikov_dichte,"Epanechnikov")

plot_data(0.8,epanechnikov_dichte,"Epanechnikov")

plot_data(1.2,epanechnikov_dichte,"Epanechnikov")