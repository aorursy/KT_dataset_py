# import libraries
import numpy as np
import matplotlib.pyplot as plt

# define constants
Kb = 1.76E-05
Kw = 1.01E-14
F  = 0.0100
gamma_H = 1.0000
gamma_OH = 1.0000
gamma_NH4 = 1.0000

# create a function to calculate the Kb based on our current values the input is a [H+] and the returned value is the calculated Kb value
# using the equations that we derived
def calcKb(x):
    H = x
    NH4 = (Kw / H) - H
    OH  = (Kw / H)
    NH3 = F - OH + H
    Kbcalc = (NH4 * gamma_NH4) * (OH * gamma_OH) / NH3
    return Kbcalc

# create a function to optimize the [H+] so we can find the right value that gives us the correct Kb value
# there is no input required and the returned value is the [H+] yielding the correct Kb
# this algorithm is very simple: it increases [H+] until our Kb value crosses the actual Kb. then we cut
# the step to 10% of the original value and change the direction. we keep doing that until we achieve a %error < 0.01%
def optimizeH():
    counter = 0
    Kbcalc = 0.000E-05
    Hinitial = 1.0000E-11
    Htest = Hinitial
    stepsize = 0.1000E-11
    percent_error = 1

    while (Kbcalc != Kb and counter < 100 and percent_error > 0.01):
        counter += 1
        Htest += stepsize

        if (calcKb(Htest) < Kb and stepsize > 0):
            stepsize *= -0.10
        if (calcKb(Htest) > Kb and stepsize < 0):
            stepsize *= -0.10
        percent_error = 100 * np.abs(calcKb(Htest) - Kb) / Kb
        
    return Htest

# given an ionic strength (a) and radius (b), we calculate the activity coefficient
def calc_gamma(a, b):
    u = a
    radius = b
    gamma = 10**(-0.51 * np.sqrt(u) / (1 + (radius * np.sqrt(u) / 305)))
    return gamma

# create a function that will output everything we have in a pretty format AND pack all of our values
# into the appropriate arrays and array elements
def pretty_print(d):
    print("[H+]: {:05.4E}".format(Hconc))
    print("[OH-]: {:05.4E}".format(OH))
    print("[NH4]: {:05.4E}".format(NH4))
    print("[NH3]: {:05.4E}".format(NH3))
    print("pH: {:05.2f}".format(pH))
    print("gammaH = {:05.4f}".format(gamma_H))
    print("gammaOH = {:05.4f}".format(gamma_OH))
    print("gammaNH4 = {:05.4f}".format(gamma_NH4))

    Harray[d] = Hconc
    NH3array[d] = NH3
    NH4array[d] = NH4
    OHarray[d] = OH
    pHarray[d] = pH
    gHarray[d] = gamma_H
    gNH4array[d] = gamma_NH4
    gOHarray[d] = gamma_OH
    
# create a function to do the actual iterations
def ready_set_go(e):
    Hconc = optimizeH()
    OH = Kw / Hconc
    NH4 = Kw / Hconc - Hconc
    NH3 = F - NH4

    ionic_strength = 0.5 * (Hconc + OH + NH4)

    gamma_H = calc_gamma(ionic_strength, 900)
    gamma_OH = calc_gamma(ionic_strength, 350)
    gamma_NH4 = calc_gamma(ionic_strength, 250)

    pH = -1 * np.log10(Hconc * gamma_H)
    
    pretty_print(e)

Harray = np.zeros(4)
NH3array = np.zeros(4)
NH4array = np.zeros(4)
OHarray = np.zeros(4)
pHarray = np.zeros(4)
gHarray = np.zeros(4)
gNH4array = np.zeros(4)
gOHarray = np.zeros(4)
# round 1
Hconc = optimizeH()
OH = Kw / Hconc
NH4 = Kw / Hconc - Hconc
NH3 = F - NH4

ionic_strength = 0.5 * (Hconc + OH + NH4)

gamma_H = calc_gamma(ionic_strength, 900)
gamma_OH = calc_gamma(ionic_strength, 350)
gamma_NH4 = calc_gamma(ionic_strength, 250)

pH = -1 * np.log10(Hconc * gamma_H)

pretty_print(0)
# round 2
Hconc = optimizeH()
OH = Kw / Hconc
NH4 = Kw / Hconc - Hconc
NH3 = F - NH4

ionic_strength = 0.5 * (Hconc + OH + NH4)

gamma_H = calc_gamma(ionic_strength, 900)
gamma_OH = calc_gamma(ionic_strength, 350)
gamma_NH4 = calc_gamma(ionic_strength, 250)

pH = -1 * np.log10(Hconc * gamma_H)

pretty_print(1)
# round 3
Hconc = optimizeH()
OH = Kw / Hconc
NH4 = Kw / Hconc - Hconc
NH3 = F - NH4

ionic_strength = 0.5 * (Hconc + OH + NH4)

gamma_H = calc_gamma(ionic_strength, 900)
gamma_OH = calc_gamma(ionic_strength, 350)
gamma_NH4 = calc_gamma(ionic_strength, 250)

pH = -1 * np.log10(Hconc * gamma_H)

pretty_print(2)
# round 4
Hconc = optimizeH()
OH = Kw / Hconc
NH4 = Kw / Hconc - Hconc
NH3 = F - NH4

ionic_strength = 0.5 * (Hconc + OH + NH4)

gamma_H = calc_gamma(ionic_strength, 900)
gamma_OH = calc_gamma(ionic_strength, 350)
gamma_NH4 = calc_gamma(ionic_strength, 250)

pH = -1 * np.log10(Hconc * gamma_H)

pretty_print(3)
# i'm calling element 0 in all arrays as the activity coefficients before doing
# anything (we started at 1.0000)

x = np.array([0, 1, 2, 3])

plt.plot(x, gHarray)
plt.show()
# here is the pH as we went through 4 iterations of the systematic treatment
print("pH from systematic treatment")
for a in pHarray:
    print("  {:05.3f}".format(a))

# we could also calculate the pH using the first approximation (Kb = x^2 / (F-x))
root = [x for x in np.roots([1, Kb, -1 * Kb * F]) if x > 0]
print("Kb = x^2 / (F-x)")
print("  {:05.3f}".format(float(14 + np.log10(root))))

# we could also calculate the pH using another approximation that F-x ~ F
approx = np.sqrt(Kb/F)
print("Kb = x^2 / F")
print("  {:05.3f}".format(14 + np.log10(approx)))
