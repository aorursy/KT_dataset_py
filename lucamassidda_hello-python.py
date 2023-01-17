# Esempio, non modificare!
print(5 / 8)

# Calcola 7 + 10
print(7 + 10)
print(1.0+2.2)
# Addizione, sottrazione
print(5 + 5)
print(5 - 5)

# Moltiplicazione, divisione, modulo ed esponenziale
print(3 * 5)
print(10 / 2)
print(18 % 7)
print(4 ** 2)
# Quanto valgono 100 Euro dopo 7 anni?
print(100*(1.1**7))
# 21°C in °F?
print(21.0*1.8 + 32.0)
print("Temperatura: " + str(21.0) + "°C")
print("Temperatura: " + str(round(21.0*1.8+32.0,1)) + "°F")
print("Temperatura: " + str(21.0) + "°C")
# 21°C in K?
print(21.0 + 273.15)
# Crea una variabile temperatura

temperatura_C = 21

# Stampa il suo valore
print("Temp: " + str(temperatura_C) +"°C")
# Crea una variabile risparmi
risparmi = 200

# Create a variable interessi
interessi = 0.10

# Create una variable anni di interesse
anni = 20

# Calcola il montante
montante = risparmi*(1+interessi)**anni

# Stampa il risultato
print("dopo " + str(anni) + " ho maturato " + str(round(montante, 2)))
# Crea una variabile risparmi
risparmi = 300

# Create a variable interessi
interessi = 0.10

# Create una variable anni di interesse
anni = 20

# Calcola il montante
montante = risparmi*(1+interessi)**anni

# Stampa il risultato
print("dopo " + str(anni) + " ho maturato " + str(round(montante, 2)))
print(risparmi)
# Crea una variabile temperatura
temperatura_C = 22

# Create due variabili per il fattore di scala a pari a 1.8 e per l'intercetta b pari a 32
a = 1.8
b = 32

# Calcola la temperatura in Fahreneit
temperatura_F = temperatura_C * a + b

# Stampa il risultato
print("Temperatura: " + str(round(temperatura_F, 1)) + "°F")

# Crea una variabile temperatura
temperatura_C = 37.8

# Create due variabili per il fattore di scala a pari a 1.8 e per l'intercetta b pari a 32
a = 1.8
b = 32

# Calcola la temperatura in Fahreneit
temperatura_F = temperatura_C * a + b

# Stampa il risultato
print("Temperatura: " + str(round(temperatura_F, 1)) + "°F")
# Calcola la temperatura_K a partire da temperatura_C
temperatura_K = temperatura_C + 273.15

# Stampa il risultato
print("Temp: " + str(round(temperatura_K, 2)) + "K")

print(round(temperatura_F, 2))
spam_amount = 0
print(spam_amount)

# Ordering Spam, egg, Spam, Spam, bacon and Spam (4 more servings of Spam)
spam_amount = spam_amount + 4

if spam_amount > 0:
    print("But I don't want ANY spam!")

viking_song = "Spam " * spam_amount
print(viking_song)
spam_amount = 0
type(spam_amount)
type(19.95)
print(min(1, 2, 3))
print(max(1, 2, 3))
print(abs(32))
print(abs(-32))
print(float(10))
print(int(3.33))
# They can even be called on strings!
print(int('807') + 1)
# K = 3/2 k T
TC = 21 # temperatura in °C
k = 1.38*10**(-23) # costante di Boltzmann
k = 1.38E-23 # costante di Boltzmann
T = TC + 273.15 # temperatura in Kelvin
K = (3.0/2.0) * k * T
print("Energia cinetica: " + str(K) + "J") # 6.09E-21 ???
N_A = 6.022E23 # Numero di Avogadro
N_A = 6.022*10**23 # Numero di Avogadro

K_mole = N_A * K # Energia di una mole di gas
print("Energia cinetica si una mole: " + str(K_mole) + "J")
# 1 kcal = 4184 J
W = 1200 # potenza dell'asciugacapelli
t = 20*60 #tempo di utilizzo

Ephon_J = W * t # l'energia é potenza per il tempo di utilizzo

print("Energia assorbita: " + str(Ephon_J) + " J")

Ephon_kcal = Ephon_J / 4184 # fattore di conversione tra Joule e kcal

print("Energia assorbita: " + str(Ephon_kcal) + " kcal")
import matplotlib.pyplot as plt
import numpy as np
alpha = np.arange(0,2*np.pi,0.1)
plt.plot(np.sin(alpha));
plt.plot(np.cos(alpha));

#plt.plot(np.tan(alpha));
# E = ?
Ephon_kWh = W /1000 * t / 3600
print("Energia assorbita: " + str(Ephon_kWh) + " kWh")
Ephon_J / Ephon_kWh
# T = ?
# S0 = 1367 W/m2
# S0 = 4 sigma T4
# sigma = 5.67 x 10-8 W·m-2·K-4
# T = 
# sigma = 5.67 x 10-8 W·m-2·K-4
# A = 0.3
# T = 
# sigma = 5.67 x 10-8 W·m-2·K-4
# A = 0.4
# T = 
# pressione in superficie

# densità dell'acqua

# accelerazione di gravità

# legge di Stevino

