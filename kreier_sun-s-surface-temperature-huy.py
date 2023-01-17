def Solar_Constant(deltaT, m = 30, r = 0.011, h = 0.1, t = 300):

    return(((m * 4.186 * deltaT)/t)/(2 * r * h))



def Temp(Solar_Constant):

    numerator = Solar_Constant * ((1.495978707*(10**11))**2)

    demominator = (5.67*(10**(-8))) * ((6.955*(10**8))**2)

    return (numerator/demominator)**(0.25)



TempChange = input("Input the predicted temperature change: ")

S = Solar_Constant(float(TempChange))

Sun = Temp(S)

print(str(int(S)) + " w/m^2 is the Solar Constant calculated from the given change in temperature.")

print(str(int(Sun)) + "K is the temperature of the Sun.")