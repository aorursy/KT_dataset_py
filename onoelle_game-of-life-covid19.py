import matplotlib.pyplot as plt
import random
from random import choice
from IPython.display import clear_output
import numpy as np
import pandas as pd
class habitant():

    def __init__(self, boundx, boundy):
        # Zufallsstartbelegung mit Abstand 20 von Rand
        self.x_value = random.randint(20,boundx-20) 
        self.y_value = random.randint(20,boundy-20)
        # 0=susceptible 1=infected 2=recovered
        self.condition = 0
        self.infectticker = 0

    
    def move(self, boundx, boundy):
        # Führt Schritte aus, bis der Pfad die angegebene Länge erreicht hat.
            x_direction = choice([1, -1])
            x_distance = choice([0, 1, 2, 3, 4])
            x_step = x_direction * x_distance
            y_direction = choice([1, -1])
            y_distance = choice([0, 1, 2, 3, 4])
            y_step = y_direction * y_distance
            
            # Berechnet den nächsten x- und y-Wert.
            self.x_value += x_step 
            self.y_value += y_step
            
            if self.x_value < 1 or self.x_value > boundx-1:
                self.x_value -= x_step
                
            if self.y_value < 1 or self.y_value > boundy-1:
                self.y_value -= y_step
    
    

population = 100
country_x = 100
country_y = 100
initial_infected = 3
time_to_recover = 100
steps = 1000
%matplotlib inline

person = [habitant(country_x, country_y) for dummy in range(population)]
for b in range(initial_infected):
    person[b].condition = 1

healthmatrix = np.zeros([country_x, country_y], dtype = int)

susceptibel_time = np.zeros([steps], dtype = int)
infected_time = np.zeros([steps], dtype = int)
recovered_time = np.zeros([steps], dtype = int)
infected = initial_infected
susceptibel = population-infected
recovered = 0


for i in range(steps):
    for j in range (population):
        xpos = person[j].x_value
        ypos = person[j].y_value
        person[j].move(country_x, country_y)
        
        if person[j].condition == 1:
                person[j].infectticker += 1
                
        if person[j].condition == 1 and person[j].infectticker >= time_to_recover:
                person[j].condition = 3
                healthmatrix[xpos, ypos] = 0
                recovered += 1
                infected -= 1
        
        if person[j].condition == 1 :
            healthmatrix[person[j].x_value, person[j].y_value] = 1
            healthmatrix[xpos, ypos] = 0
            
        if person[j].condition == 0 and (healthmatrix[person[j].x_value, person[j].y_value] == 1 or healthmatrix[xpos, ypos] == 1):
            person[j].condition = 1  
            infected += 1
            susceptibel -= 1
            
        if person[j].condition == 0 :
            colour = "green"
        elif person[j].condition == 1:
            colour = "red"
        else:
            colour = "blue"
        
        fig_size = plt.rcParams["figure.figsize"]
        fig_size[0] = 12
        fig_size[1] = 4
        plt.rcParams["figure.figsize"] = fig_size
            
        plt.axis( [0,country_x,0,country_y] ) 
        sub1 = plt.subplot(1, 2, 1)
        sub1.scatter(person[j].x_value, person[j].y_value, c=colour, s=100)
    
    susceptibel_time[i] = susceptibel
    infected_time[i] = infected
    recovered_time[i] = recovered
    
    df = pd.DataFrame(susceptibel_time,columns=["susceptibel"])
    df['infected']=infected_time
    df['recovered']=recovered_time
    df['step'] = range(1, len(df) + 1)

    sub2 = plt.subplot(1, 2, 2)
    df.plot(ax=sub2, x = 'step', y = ["infected","recovered","susceptibel"],kind='line', title = 'course of infection', legend = False)
    plt.gca().get_lines()[0].set_color("red")
    plt.gca().get_lines()[1].set_color("blue")
    plt.gca().get_lines()[2].set_color("green")
    
    clear_output(wait=True)
    plt.pause(0.01)
    
plt.show()
    
df.plot(x = 'step', y = ["infected","recovered","susceptibel"],kind='line', title = 'Number cases: red=Infected, green=susceptibel, blue=recovered ', legend = False)
plt.gca().get_lines()[0].set_color("red")
plt.gca().get_lines()[1].set_color("blue")
plt.gca().get_lines()[2].set_color("green")
df.head(5)
from scipy.integrate import odeint
from scipy import integrate, optimize
def SIR_model(y,t,beta,gamma):
    S, I, R = y
    N = population
    dS_dt = -1*beta*I*S/N 
    
    dI_dt = (beta*I*S/N) - gamma*I
    
    if t > time_to_recover : 
        dR_dt = gamma*I
    else : 
        dR_dt = 0
    
    return ([dS_dt, dI_dt, dR_dt])

def fit_odeint(x, beta, gamma):
    return integrate.odeint(SIR_model, (S0, I0, R0), x, args=(beta, gamma))[:,1]
xdata = df.step
ydata = df.infected
xdata = np.array(xdata, dtype=float)
ydata = np.array(ydata, dtype=float)

S0 = population-initial_infected
I0 = initial_infected
R0 = 0
y = S0, I0, R0

popt, pcov = optimize.curve_fit(fit_odeint, xdata, ydata)
fitted = fit_odeint(xdata, *popt)
plt.plot(xdata, ydata, 'o')
plt.plot(xdata, fitted)
plt.title("Fit of SIR model for Germany infected cases")
plt.ylabel("Population infected")
plt.xlabel("Days")
plt.show()
print("Optimal parameters: beta =", popt[0], " and gamma = ", popt[1])
#defining initial conditions

N = population
S00 = population-initial_infected
I00 = initial_infected
R00 = 0.0
#bta = 0.015   # as infection time is 100, multiply beta*100 to get Relication Rate (R0)
bta = popt[0]
gmma = popt[1]

t = np.linspace(0,1000,1000)

sol = odeint(SIR_model,[S00,I00,R00],t,args = (bta,gmma))
sol = np.array(sol)
#plotting results

new_susp=population-sol[:,1]-sol[:,2]

plt.figure(figsize=(12,8))
plt.plot(t, new_susp,label = "S(t)")
plt.plot(t, sol[:,1],label = "I(t)")
plt.plot(t, sol[:,2],label = "R(t)")
plt.plot(t, infected_time,label = "I(t) mdel")
plt.plot(t, susceptibel_time,label = "S(t) mdel")
plt.plot(t, recovered_time,label = "R(t) mdel")
plt.legend()
plt.show()
Rnull = bta*time_to_recover
print (Rnull)
def deriv(y, t, N, beta, gamma, delta):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - delta * E
    dIdt = delta * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

# calculate optimized beta and gamma (tbd)
def fit_odeint2(x, beta, gamma, delta):
#    return integrate.odeint(SIR_model, (S0, I0, R0), x, args=(beta, gamma))[:,1]
    return integrate.odeint(deriv, (S0, E0, I0, R0), x, args=(N, beta, gamma, delta))[:,1]
N = population
beta = 0.6  # infected person infects 1 other person per day
D = 40.0 # infections lasts four days
gamma = 1.0 / D
delta = 1.0 / 100.0  # incubation period of three days

#D = time_to_recover
#gamma = 1.0/D


# S0, I0, R0, E0 = 999, 1, 0, 0  # initial conditions: one infected, rest susceptible
S0 = population-initial_infected
I0 = 0.0
R0 = 0.0
E0 = initial_infected
t = np.linspace(0, 1000, 1000) # Grid of time points (in days)
y0 = S0, E0, I0, R0 # Initial conditions vector

# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma, delta))
S, E, I, R = ret.T
plt.figure(figsize=(12,8))
plt.plot(t, S,label = "S")
plt.plot(t, I,label = "I")
plt.plot(t, R,label = "R")
plt.plot(t, E,label = "E")
plt.legend()
plt.show()