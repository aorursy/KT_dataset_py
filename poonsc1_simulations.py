# We first begin by creating a function to initialise these values

def SIRsim(S0, I0, R0, beta, gamma, maxT=200): # maxT = time allowed for simulation to run
    N = S0 + I0 + R0
    
    # We initialise a list for each variable, with each element corresponding to each time period. These will be used for plots.
    Time = [0] * maxT
    S = [0] * maxT
    I = [0] * maxT
    R = [0] * maxT
    # These values will be filled as we move along
    
    # First values of each list shall be speicifed by the input parameters
    Time[0] = 0
    S[0] = S0
    I[0] = I0
    R[0] = R[0]
    
    # We create a for loop, to calculate values for each variable at time T given the formulas in the paper.
    # Within this for loop, the relationship for each variable w.r.t time is given on the RHS.
    for t in range(maxT-1):
        Time[t+1] = t+1
        S[t+1] = round(S[t] - beta*((S[t]*I[t])/N))
        I[t+1] = round(I[t] + beta*((S[t]*I[t])/N) - gamma*I[t])
        R[t+1] = N - S[t+1] - I[t+1]
     
    return Time, S, I, R

    
from matplotlib import pyplot as plt

Time, S, I, R = SIRsim(900, 100, 0, 0.25, 0.1) # For first simulation, we input these values

plt.plot(Time, S)
plt.plot(Time, I)
plt.plot(Time, R)
plt.legend(['S', 'I', 'R'])
plt.show();
# For example, would a more people spreading a message of less infectivity be better?

Time, S, I, R = SIRsim(500, 500, 0, 0.10, 0.1) # For first simulation, we input these values

plt.plot(Time, S)
plt.plot(Time, I)
plt.plot(Time, R)
plt.legend(['S', 'I', 'R'])
plt.show();
# For example, would a less people but a message with longer-lasting effect be better?

Time, S, I, R = SIRsim(950, 50, 0, 0.25, 0.05) # For first simulation, we input these values

plt.plot(Time, S)
plt.plot(Time, I)
plt.plot(Time, R)
plt.legend(['S', 'I', 'R'])
plt.show();
