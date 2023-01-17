import pandas as pd #Processamento e manipulação de dados.
Pr = 300 #Pressão estática do reservatório, [kgf/cm²].
Psat = 180 #Pressão de saturação, [kgf/cm²].
P1 = 250 # 1° Pressão no teste de produção, [kgf/cm²].
Q1 = 1000 # 1° Vazão no teste de produção, [m³/dia].
J = Q1/(Pr-P1) #Produtividade do reservatório[m³cm²/Kgfdia]
ΔP = 20 # Queda de Pressão, [Kgf/cm²]
def Qmon(J, Pr, P): # Função para calculo da vazão de fluxo monofásico.
    Qmon = J*(Pr-P)
    return Qmon

def Qmax (J, Psat, Qsat): # Função para calculo da vazão máxima.
    Qmax = J*Psat/(1.8)+Qsat
    return Qmax

def Qbif (P, Psat, Qmax, Qs): # Função para calculo da vazão de fluxo bifásico, equação de Vogel.
    Qbif = (1-0.2*(P/Psat)-0.8*(P/Psat)**2)*(Qmax-Qs)+Qs
    return Qbif
Qsat = Qmon(J, Pr, Psat) # Vazão para pressão de saturação, [m³/dia].

Qmax = Qmax(J, Psat, Qsat) # Vazão máxima [m³/dia].
def IPR (Psat, J, Pr, Qmax, Qsat, ΔP): #Função para calculo de pressões e vazões.
    PxQ = [[Pr,0]]
    P = Pr - ΔP
    while P > 0:
        if P >= Psat:
            PxQ.append([P,Qmon(J, Pr, P)])
            P = P - ΔP
        if P < Psat:
            PxQ.append([P,Qbif(P, Psat, Qmax, Qsat)])
            P = P - ΔP
    if P <=0:
        P = 0
        PxQ.append([P, Qmax])
    PxQ = pd.DataFrame(data = PxQ, columns = ["Pressão [kgf/cm²]","Vazão [m³/dia]"])
    return PxQ
PxQ = IPR (Psat, J, Pr, Qmax, Qsat, ΔP) #Calculo do IPR.
PxQ
PxQ.plot( x = "Vazão [m³/dia]", y = "Pressão [kgf/cm²]", title = "Curva IPR - Combinada", label = "IPR",xlabel = "Vazão [m³/dia]", ylabel = "Pressão [kgf/cm²]")# Plotagem do IPR