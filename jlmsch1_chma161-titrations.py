# import the important libraries

import numpy as np                  # gives us some math functions and array operations

import matplotlib.pyplot as plt     # gives us the ability to plot our data

import matplotlib as mpl
# how many elements should be in our list? 141 will give us a list of pH values from 0.00 - 14.00 in

# 0.10 mL increments. that's a good start but we'll change this later.

mono_array_size = 141



# i'm using the solutions shown on page 255-256 of the Harris book. These are opposite of what we have

# done in class. we usually pick a volume and then calculate the pH at that volume. THESE EQUATIONS

# USE THE PH AS INPUT AND THEN CALCULATE THE VOLUME - OPPOSITE!!! it's not intuitive for us but we

# do it because it's easier to simulate.

# we need to make a list of pH values where we would like to calculate the volume. the following 

# line will create a list of numbers that go from 0.00 - 14.00 in 0.10 pH increments

mono_pH_array = np.linspace(0.00, 14.00, mono_array_size)

#mono_pH_array = np.array([3.23, 5.66, 6.94, 9.82, 11.96, 12.64])



# we should also create an empty list to store our volumes once they are calculated.

# technically, this list isn't empty but rather contains a series of zeros. that makes it the same

# size as the pH_array list above

mono_vol_array = np.zeros(mono_array_size)

#mono_vol_array = np.array([3.23, 5.66, 6.94, 9.82, 11.96, 12.64])



# we also need to define all of the constants for our particular system - this is the titration of MES

# with NaOH that we did in class

mono_conc_base = 0.7692 # concentration of base (NaOH; titrant) in molar



mono_conc_acid = 0.5000 # concentration of acid (MES) in molar

mono_vol_acid = 50.00   # volume of acid (MES) in mL



mono_pKa = 6.15         # pKa of the acid (MES)

mono_Ka = 10**-mono_pKa # Ka of the acid (MES)

Kw = 1.01e-14           # Kw of water at 25 degrees C



# this function will be used to run through our list of pH values and will calculate the

# corresponding volume for that specific pH

# NOTE: this is only a model. while you can calculate a volume for each pH value in the list,

# it's up to you (the expert) to determine if that number makes sense. to give you an example,

# it's impossible to obtain a pH above 13.00 with a solution of 0.1000 M NaOH; the calculated

# volumes above pH 13.00 are garbage so I filter them out.

def mono_calculate_volume(pH):

    H_plus = 10**-pH

    OH_minus = Kw / H_plus

    alpha_A_minus = mono_Ka / (H_plus + mono_Ka)

    phi = (alpha_A_minus - (H_plus - OH_minus)) / (1 + (H_plus - OH_minus) / mono_conc_base)

    vol_base = phi * mono_conc_acid * mono_vol_acid / mono_conc_base

    if (vol_base > 0 and vol_base < 50):

        return vol_base

    else:

        return np.nan



index = 0



# step through all of the values in the pH list and calculate the volume, then store the value

for a in mono_pH_array:

    mono_vol_array[index] = mono_calculate_volume(a)

    index += 1



font = {'family' : 'normal',

        'weight' : 'bold',

        'size'   : 22}



mpl.rc('font', **font)

    

# make a pretty picture

plt.figure(figsize=(15,7.5))

plt.plot([0.00, 16.25, 32.50, 35.00], [3.23, 6.15, 9.81, 12.45], 'ro', markersize=20)

plt.plot(mono_vol_array, mono_pH_array, linewidth=4)

plt.xlabel('vol [mL]')

plt.ylabel('pH')

plt.grid(True)

plt.title('Titration of MES with KOH')

plt.minorticks_on()

plt.axis([0.00, 50.00, 0.00, 14.00]) # YOU SHOULD CHANGE THESE NUMBERS TO "ZOOM IN" TO YOUR PEAK (XMIN, XMAX, YMIN, YMAX)

plt.show()
# we need to setup empty lists so we have a place to store the results of our first derivatives

mono_vol_array_1st = np.zeros(mono_array_size)

mono_pH_array_1st = np.zeros(mono_array_size)



# here is where we actually calculate the first derivative. this is EXACTLY what is shown on

# page 244 of the textbook. this is EXACTLY what you have to do in lab this week!

for index in range(0, mono_array_size - 1, 1):

    mono_vol_array_1st[index] = (mono_vol_array[index] + mono_vol_array[index + 1]) / 2

    mono_pH_array_1st[index] = (mono_pH_array[index + 1] - mono_pH_array[index]) / (mono_vol_array[index + 1] - mono_vol_array[index])



# make a pretty picture

plt.figure(figsize=(15,7.5))

plt.plot(mono_vol_array_1st, mono_pH_array_1st)

plt.xlabel('vol [mL]')

plt.ylabel('first derivative')

plt.grid(True)

plt.title('Titration of MES with NaOH :: First derivative')

plt.minorticks_on()

plt.axis([32.00, 34.00, 0.00, 40.00]) # YOU SHOULD CHANGE THESE NUMBERS TO "ZOOM IN" TO YOUR PEAK (XMIN, XMAX, YMIN, YMAX)

plt.show()
# we need to setup empty lists so we have a place to store the results of our second derivatives

mono_vol_array_2nd = np.zeros(mono_array_size)

mono_pH_array_2nd = np.zeros(mono_array_size)



# here is where we actually calculate the first derivative. this is EXACTLY what is shown on

# page 244 of the textbook. this is EXACTLY what you have to do in lab this week!

for index in range(0, mono_array_size - 1, 1):

    mono_vol_array_2nd[index] = (mono_vol_array_1st[index] + mono_vol_array_1st[index + 1]) / 2

    mono_pH_array_2nd[index] = (mono_pH_array_1st[index + 1] - mono_pH_array_1st[index]) / (mono_vol_array_1st[index + 1] - mono_vol_array_1st[index])



# make a pretty picture

plt.figure(figsize=(15,7.5))

plt.plot(mono_vol_array_2nd, mono_pH_array_2nd)

plt.xlabel('vol [mL]')

plt.ylabel('second derivative')

plt.grid(True)

plt.title('Titration of MES with NaOH :: Second derivative')

plt.minorticks_on()

plt.axis([32.00, 34.00, -1500.00, 1500.00]) # YOU SHOULD CHANGE THESE NUMBERS TO "ZOOM IN" TO YOUR PEAK (XMIN, XMAX, YMIN, YMAX)

plt.show()
# how many elements should be in our list? 141 will give us a list of pH values from 0.00 - 14.00 in

# 0.10 mL increments. that's should be good.

poly_array_size = 141



# i'm using the solutions shown on page 255-256 of the Harris book. These are opposite of what we have

# done in class. we usually pick a volume and then calculate the pH at that volume. THESE EQUATIONS

# USE THE PH AS INPUT AND THEN CALCULATE THE VOLUME - OPPOSITE!!! it's not intuitive for us but we

# do it because it's easier to simulate.

# we need to make a list of pH values where we would like to calculate the volume. the following 

# line will create a list of numbers that go from 0.00 - 14.00 in 0.10 pH increments

poly_pH_array = np.linspace(0.00, 14.00, poly_array_size)



# we should also create an empty list to store our volumes once they are calculated.

# technically, this list isn't empty but rather contains a series of zeros. that makes it the same

# size as the pH_array list above

poly_vol_array = np.zeros(poly_array_size)



# we also need to define all of the constants for our particular system - this is a titration of citric acid

# with NaOH. it's similar to what you will do in lab next week.

poly_conc_base = 0.16 # concentration of base (NaOH; titrant) in molar



poly_conc_acid = 0.0400 # concentration of acid (citric) in molar

poly_vol_acid = 50.00   # volume of acid (citric) in mL



poly_pKa1 = 2     # pKa1 of the acid

poly_pKa2 = 6     # pKa2 of the acid

poly_pKa3 = 10     # pKa3 of the acid

poly_Ka1 = 10**-poly_pKa1      # pKa1 of the acid

poly_Ka2 = 10**-poly_pKa2      # pKa2 of the acid

poly_Ka3 = 10**-poly_pKa3      # pKa3 of the acid

Kw = 1.01e-14           # Kw of water at 25 degrees C



# this function will be used to run through our list of pH values and will calculate the

# corresponding volume for that specific pH

# NOTE: this is only a model. while you can calculate a volume for each pH value in the list,

# it's up to you (the expert) to determine if that number makes sense. to give you an example,

# it's impossible to obtain a pH of 13.00 with a solution of 0.1000 M NaOH; the calculated

# volumes are garbage so I filter them out.

def poly_calculate_volume(pH):

    H_plus = 10**-pH

    OH_minus = Kw / H_plus

    poly_D = ((H_plus**3) + (H_plus**2 * poly_Ka1) + (H_plus * poly_Ka1 * poly_Ka2) + (poly_Ka1 * poly_Ka2 * poly_Ka3))

    alpha_H2A_minus = (H_plus**2 * poly_Ka1) / poly_D

    alpha_HA_2minus = (H_plus * poly_Ka1 * poly_Ka2) / poly_D

    alpha_A_3minus = (poly_Ka1 * poly_Ka2 * poly_Ka3) / poly_D

    phi = (alpha_H2A_minus + 2 * alpha_HA_2minus + 3 * alpha_A_3minus - (H_plus - OH_minus)) / (1 + (H_plus - OH_minus) / poly_conc_base)

    vol_base = phi * poly_conc_acid * poly_vol_acid / poly_conc_base

    if (vol_base > 0 and vol_base < 50):

        return vol_base

    else:

        return np.nan



index = 0



# step through all of the values in the pH list and calculate the volume, then store the value

for a in poly_pH_array:

    poly_vol_array[index] = poly_calculate_volume(a)

    index += 1



# make a pretty picture

plt.figure(figsize=(15,7.5))

plt.plot(poly_vol_array, poly_pH_array)

plt.xlabel('vol [mL]')

plt.ylabel('pH')

plt.grid(True)

plt.title('Titration of citric acid with NaOH')

plt.minorticks_on()

plt.axis([0.00, 50.00, 0.00, 14.00])

plt.show()
# we need to setup empty lists so we have a place to store the results of our first derivatives

poly_vol_array_1st = np.zeros(poly_array_size)

poly_pH_array_1st = np.zeros(poly_array_size)



# here is where we actually calculate the first derivative. this is EXACTLY what is shown on

# page 244 of the textbook. this is EXACTLY what you have to do in lab next week!

for index in range(0, poly_array_size - 1, 1):

    poly_vol_array_1st[index] = (poly_vol_array[index] + poly_vol_array[index + 1]) / 2

    poly_pH_array_1st[index] = (poly_pH_array[index + 1] - poly_pH_array[index]) / (poly_vol_array[index + 1] - poly_vol_array[index])



# make a pretty picture

plt.figure(figsize=(15,7.5))

plt.plot(poly_vol_array_1st, poly_pH_array_1st)

plt.xlabel('vol [mL]')

plt.ylabel('first derivative')

plt.grid(True)

plt.title('Titration of citric acid with NaOH :: First derivative')

plt.minorticks_on()

plt.axis([0.00, 50.00, 0.00, 2.00]) # YOU SHOULD CHANGE THESE NUMBERS TO "ZOOM IN" TO YOUR PEAKS (XMIN, XMAX, YMIN, YMAX)

plt.show()
# we need to setup empty lists so we have a place to store the results of our second derivatives

poly_vol_array_2nd = np.zeros(poly_array_size)

poly_pH_array_2nd = np.zeros(poly_array_size)



# here is where we actually calculate the first derivative. this is EXACTLY what is shown on

# page 244 of the textbook. this is EXACTLY what you have to do in lab this week!

for index in range(0, poly_array_size - 1, 1):

    poly_vol_array_2nd[index] = (poly_vol_array_1st[index] + poly_vol_array_1st[index + 1]) / 2

    poly_pH_array_2nd[index] = (poly_pH_array_1st[index + 1] - poly_pH_array_1st[index]) / (poly_vol_array_1st[index + 1] - poly_vol_array_1st[index])



# make a pretty picture

plt.figure(figsize=(15,7.5))

plt.plot(poly_vol_array_2nd, poly_pH_array_2nd)

plt.xlabel('vol [mL]')

plt.ylabel('second derivative')

plt.grid(True)

plt.title('Titration of citric acid with NaOH :: Second derivative')

plt.minorticks_on()

plt.axis([0.00, 50.00, -3.00, 3.00]) # YOU SHOULD CHANGE THESE NUMBERS TO "ZOOM IN" TO YOUR PEAK (XMIN, XMAX, YMIN, YMAX)

plt.show()