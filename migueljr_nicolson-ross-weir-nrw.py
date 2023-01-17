from __future__ import division

import os 

import numpy as np

import matplotlib.pyplot as plt







#-----------------Graphic Size------------------

#Plot size

x = 7

y = 7

#x and y axis size

fonte = 15



#skip point

skip_point = 30

#--------------Read the file-------------

arq = open("../input/permitivity-and-permeability-through-sparameters/par_s_teflon_5mm.txt", 'r')

ler = arq.readlines()

arq.close()

# ------------------Guide Parameters-------------

d = 5.0e-3 #[m] thickness sample

a = 22.86e-3 #[m] larger waveguide base (X-Band)

offset = 9.76e-3 #1/4 lambda thickness





L1 = 0.0e-3 #[m] Reference plan door 1

L2 = offset - d #[m] Reference plan door 2







#properties:

c = 2.998e8 #[m/s] speed of light

u0=4*np.pi*1e-7 # Vacuum permeability

freq_corte = 6.56e9 #[Hz] cut frequency (X-Band)

onda_cut = c/freq_corte #[m] cut lambda
#-------------------------VETORES-1-------------------		

F=[] #frequencia [Hz] DE CALCULO

F_grafic=[] #FREQUENCIA PARA PLOTAR EM [GHz]



s11r=[] #real

s11i=[] #imag

s21r=[] #real

s21i=[] #imag



s11=[] # real + j imag

s21=[] # real + j imag



s11c=[] # real + j imag (adjusted)

s21c=[] # real + j imag (adjusted)

#---------------Organize S-Parameters---------



ler1_col=1 #S11r

ler2_col=2 #S11i

ler3_col=5 #S21r

ler4_col=6 #S21i



'''

#If you want to use the S12 and S22 I have to change the:

#s11c_colocar = R2*R2*s11_colocar

#s21c_colocar = R2*R1*s21_colocar



ler1_col=7 #S22r

ler2_col=8 #S22i

ler3_col=3 #S12r

ler4_col=4 #S12i

'''
for i in range(1,len(ler)):   

    dados = ler[i].split(',')

    i = i-1



    #Ler frequencia

    f_colocar = float(dados[0]) #Hz



    F.append(f_colocar)



    #CONVERTER FREQUÊNCIA DE Hz PARA GHz

    F_grafic.append(f_colocar/1e9)

    

    #---------Organizar os PAR-S---------------------

    s11r.append(float(dados[ler1_col]))#real

    s11i.append(float(dados[ler2_col]))#imag

    s21r.append(float(dados[ler3_col]))#real

    s21i.append(float(dados[ler4_col]))#imag



    s11_colocar =s11r[i]+1j*s11i[i] # real + j imag

    s21_colocar =s21r[i]+1j*s21i[i] # real + j imag



    s11.append(s11_colocar) #add vetor s11

    s21.append(s21_colocar) #add vetor s21

    #--------------------------------------------------



    #lambda zero = comprimento de onda no vacuo

    onda = c/F[i] # [m]



    #Constante de propagacao da onda no espaco livre

    gama0 = (2j*np.pi)*np.sqrt((1.0)/(onda**(2.0))-(1.0)/(onda_cut**(2.0)))



    #Coeficiene para Ajustar os Planos de Referencia da porta 1 e 2

    R1 = np.exp(1*gama0*L1) #constantes

    R2 = np.exp(1*gama0*L2) #constantes



    #Ajustar S11 e S21

    s11c_colocar = R1*R1*s11_colocar

    s21c_colocar = R2*R1*s21_colocar



    '''

    #LEMBRE-SE DE ALTERAR QUANDO FOR USAR s12 e S22

    #s11c_colocar = R2*R2*s11_colocar #S22

    #s21c_colocar = R2*R1*s21_colocar #S12

    '''



    #Add S11 e S21 Corrigido

    s11c.append(s11c_colocar) #s11 novo

    s21c.append(s21c_colocar) #s21 novo
#--- Modulo de S11 e S21 ------





S11_mod_c=[] #Experimental Corrigido

S21_mod_c=[] #Experimental Corrigido 



S11_mod=[] # Experimental sem corrigir

S21_mod=[] # Experimental sem corrigir







for i in range(0,len(s11)):



    S11_mod_c.append(abs(s11c[i]))# Exp Corrigido

    S21_mod_c.append(abs(s21c[i]))# Exp Corrigido



    S11_mod.append(abs(s11[i])) #Exp sem corrigir

    S21_mod.append(abs(s21[i])) #Exp sem corrigir
#Plot S11 e S21 Linear Mag



#fig=plt.figure(num=1,figsize=(x,y))



#plt.title("Linear Mag: Ajuste x sem Ajuste")

plt.title("Linear Mag - Medidos e corrigidos",fontsize = fonte)



plt.plot(F_grafic,S11_mod_c,'r-',label ="s11_Com_Ajuste",alpha=0.5,markevery=skip_point)

plt.plot(F_grafic,S21_mod_c,'b-',label="s21_Com_Ajuste",alpha=0.5,markevery=skip_point)



#plt.plot(F_grafic,S11_mod,'r-', linewidth=2,label ="s11_Sem_Ajuste")

#plt.plot(F_grafic,S21_mod,'b-', linewidth=2,label="s21_Sem_Ajuste")



plt.xlim(8.2,12.4)

plt.ylim(0,1.4)

plt.xlabel("Freq(GHz)",fontsize = fonte)

plt.ylabel("S11,S21 Linear(a.u)",fontsize = fonte)



#plt.legend(loc ='center right').get_frame().set_facecolor('0.95')

plt.legend(loc =1,fontsize = fonte).get_frame().set_facecolor('0.95')



#plt.savefig(u'Grafico_1.jpg')

plt.show()
#----------------------VETORE-parte 2---------------------



s11_ph =[] #EXP Fase CORRIGIDA

s21_ph =[] #EXP Fase CORRIGIDA



s11_db =[] #S11 EXP em dB CORRIGIDO

s21_db =[] #S21 EXP em dB CORRIGIDO



sum = [] # somatório -> sum = |s11|**2 + |s21|**2

A =[] #Absorbance

R = [] #Reflectance

TRANS = [] #Transmitance


for n in range(0,len(F)):





    #------------------- dB EXP--------------

    #Transformar S11  S21 para dB

    s11_db_calc = 20*np.log10(abs(s11c[n]))

    s21_db_calc = 20*np.log10(abs(s21c[n]))



    #add no vetor DB

    s11_db.append(s11_db_calc)

    s21_db.append(s21_db_calc)

    #--------------------------------------------



    



    #----------------FASE EXP--------------------

    #Calcular fase em radianos (conta basica de vetor)



    #Essa Funcao é usada em números reais imaginários contínuos...

    s11_ph_calc = np.arctan2(s11c[n].imag,s11c[n].real) # Para numeros complexos usar a sintaxe np.tan2(imag,real)

    s21_ph_calc = np.arctan2(s21c[n].imag,s21c[n].real) # Para numeros complexos usar a sintaxe np.tan2(imag,real)



    # Essa funcao é usada para numeros REAIS (ISOLADAS) 

    # s11_ph_calc = np.arctan(s11c[n].imag/s11c[n].real) 

    #s21_ph_calc = np.arctan(s21c[n].imag/s21c[n].real)



    #Converter rad para graus

    s11_ph_calc_grau = 360.0*s11_ph_calc/(2.0*np.pi) 

    s21_ph_calc_grau = 360.0*s21_ph_calc/(2.0*np.pi) 



    #add no vetor da phase

    s11_ph.append(s11_ph_calc_grau)

    s21_ph.append(s21_ph_calc_grau)

    #-----------------------------------------





    #------- ABSORBANCE =1 - TRANSMITANCE - REFLECTANCE-----

    reflectance = abs(s11c[n])**2

    transmitance = abs(s21c[n])**2

    absorvance = 1.0 - reflectance - transmitance



    #add vetor A,TRANS and R

    A.append(absorvance)

    TRANS.append(transmitance)

    R.append(reflectance)

    #-----------------------------------------------------------





    #--------------SUM-----------------------------------------

    #Isso mostra as perdas

    soma = reflectance + transmitance  #+ absorvance #resultado soma = 1 



    #add vetor sum

    sum.append(soma)

    #----------------------------------------------------------
#Plot Modulo em DB

#fig=plt.figure(num=1,figsize=(x,y))



plt.plot(F_grafic,s11_db,'ob',label ='s11_experimental',alpha=0.5,markevery=skip_point)

plt.plot(F_grafic,s21_db,"or",label = 's21_experimental',alpha=0.5,markevery=skip_point)



plt.ylim(-30,2)

plt.xlim(8.2,12.4)

plt.xlabel("Freq(GHz)",fontsize = fonte)

plt.ylabel("S11,S21 (dB)",fontsize = fonte)

plt.title("S11 e S21 em dB",fontsize = fonte)

#plt.legend().get_frame().set_facecolor('0.95')

#"upper left"

plt.legend(loc ='best',fontsize = fonte).get_frame().set_facecolor('0.95')

#plt.savefig(u'Grafico_2.jpg')

plt.show()

#Plot Phase em Grau

#fig=plt.figure(num=1,figsize=(x,y))



plt.plot(F_grafic,s11_ph,'g-',label ='s11_phase')

plt.plot(F_grafic,s21_ph,'b-',label='s21_phase')



#plt.plot(F_grafic,s11_ph_t,'c-', linewidth=2,label ="s11_Phase_Teorico")

#plt.plot(F_grafic,s21_ph_t,'r-', linewidth=2,label="s21_Phase_Teorico")



plt.xlim(8.2,12.4)

plt.ylim(-200,200)

plt.xlabel("Freq(GHz)",fontsize = fonte)

plt.ylabel("Fase(Graus)",fontsize = fonte)

plt.legend(loc='best',fontsize = fonte).get_frame().set_facecolor('0.95')

plt.title("FASE",fontsize = fonte)

#plt.savefig(u'Grafico_3.jpg')

plt.show()

#Plot aborbance, refletance and transmitance

#fig=plt.figure(num=1,figsize=(x,y))



plt.plot(F_grafic,A,'-b',label ='Absorbance')

plt.plot(F_grafic,TRANS,'-r',label ='Transmitance')

plt.plot(F_grafic,R,'-g',label ='Reflectance')

plt.ylim(-0.1,1.1)

plt.xlim(8.2,12.4)

plt.xlabel("Freq(GHz)",fontsize = fonte)

plt.ylabel("A,R,T(a.u)",fontsize = fonte)

plt.title("Absorbance x Reflectance x Transmitance",fontsize = fonte)

#plt.title("Absorbance, Reflectance and Transmitance")

plt.legend(loc='best',fontsize = fonte).get_frame().set_facecolor('0.95')

#plt.savefig(u'Grafico_4.jpg')

plt.show()

#Plot Sum

#fig=plt.figure(num=1,figsize=(x,y))

#plt.plot(F_grafic,sum,'-b',label ='SOMA DE S11 e S21')

plt.plot(F_grafic,sum,'-b',label ='Perda')

plt.ylim(0,1.2)

plt.xlim(8.2,12.4)

plt.xlabel("Freq(GHz)",fontsize = fonte)

plt.ylabel("(a.u)",fontsize = fonte)

plt.title("Perda", fontsize = fonte)

plt.legend(fontsize = fonte).get_frame().set_facecolor('0.95')

#plt.savefig(u'Grafico_5.jpg')

plt.show()
#-----------Vetores - parte 3------------------------



T =[] #coeficiente de transmissao dos Par-S



R =[] #coeficiente de reflexao dos Par-S



Cvetor =[] #coeficiente de reflexao dos Par-S



Z_nrw = []

#----------Cálculo de e and u ----------------------



er_r = [] #permissividade real

er_i = [] #'''''''''''''' imag

ur_r = [] #permeabilidade real

ur_i = [] #'''''''''''''' imag



er_abs=[] #permissividade modulo

ur_abs=[] #permeabilidade modulo

#VETOR - parte 3



#---------------------Método 1- Zin -------------------------------

Zin =[] #Zin Teorico com CURTO



R_Zin =[] #Coeficiente de Reflexao com CURTO TEORICO



#---------------------Método 2 - NIST NRW -------------------------



S11_r=[]
#Parâmetros do Guia

dL = 0e-3

L = d

L1  = offset - d -dL





"""

Calcular:



-Coeficiente de Transmissão

-Coeficiente de Reflexão

-Permissividade real e imaginária

-Permeabilidade real e imaginária

"""

for n in range(0,len(F)):

    

    #frequencia

    f = F[n]



    #lambda zero = comprimento de onda no vacuo

    onda = c/(f) # [m]



   



    # --------------Cálculo do coeficiente de REFLEXÃO-----------

    #Valor V1, V2, X

    V1 = s21c[n] + s11c[n]

    V2 = s21c[n] - s11c[n]

    X = (1-V1*V2)/(V1-V2)

    #X = ((s11c[n])**(2.0)-(s21c[n])**(2.0)+1)/(2*s11c[n]) 



    #Coeficiente de reflexao Par-S

    #positivo

    r_p = X + np.sqrt(X**2-1)

    #negativo

    r_n = X - np.sqrt(X**2-1)

    #condicao para valor do sinal do coeficiente de reflexao

    if r_p < 1:

        sinal = 1

    elif r_n <= 1:

        sinal =-1

    #coeficiente de reflexao com sinal ok

    r = X+sinal*(np.sqrt(X**2-1))



    R.append(abs(r))

    #----------------------------------------------------------

    

    

    # -----------------Cálculo do Coeficiente de TRANSMISSÃO ------

    #Método 1 - Livro NIST

    t = (s11c[n] + s21c[n] - r)/(1 -(s11c[n] + s21c[n])*r)

    

    T.append(abs(t))

    

    #----------------------------------------------------------------

    



    #constante P

    P2=-((1.0)/(2*np.pi*d)*np.log(1.0/t))**2

    P=1.0/np.sqrt(P2)





    #constante Lamb

    lamb = np.sqrt((1.0)/(onda)**(2)-(1.0)/(onda_cut)**(2))



    #impedancia NRW (z_nrw) #IGUAL AO Z CALCULADO COM PAR-S (z_m)

    z_nrw = (1+r)/(1-r)

    Z_nrw.append(z_nrw.real) #somente real





    #Permeabilidade NRW

    ux = z_nrw/(P*lamb)

    



    ur_r.append(ux.real)

    ur_i.append(ux.imag)

    ur_abs.append(abs(ux))





    #Permissividade NRW

    ex = ((onda)**(2)/ux)*((1.0)/(onda_cut)**2-((1)/(2*np.pi*d)*np.log(1.0/t))**2)

    #ex = ex.real - 1j*ex.imag





    er_r.append(ex.real)

    #er_i.append(ex.imag)

    er_i.append(-1*ex.imag) # Fiz isso para ficar certo, ficar positivo no gráfico...

    er_abs.append(abs(ex))

    

    

    

    

    

    # -------------- SIMULAR S11 + SHORT-CORT ------------------ 



    # Método 1 -> Zin - Refletividade (S11+CURTO)



    

    #Zin e Refletividade (S11 + curto)

    zin = (ux/ex)**(1.0/2.0)*np.tanh((2j*np.pi*d/onda)*((ux*ex)**(1.0/2.0)))

    

    #Coeficiente de Reflexao com curto

    C_curto = (zin-1)/(zin+1)

    

    #db = -20*np.log10(abs(C_curto)) #Em dB (tem o menos na frente pois o e.imaginário é positivo)

    #R_Zin.append(db)

    

    R_Zin.append(abs(C_curto)) #LINEAR MAG



    #-----------------------------------------------------

    

    #Método 2 -> Livro do NIST



    #Constante de propagacao da onda no espaco livre

    gama0 = (2j*np.pi)*np.sqrt((1.0)/(onda**(2.0))-(1.0)/(onda_cut**(2.0)))

    #Constante de Propagação da onda no Material

    gamax = (2j*np.pi/onda)*np.sqrt(ex*ux-(onda**2.0)/(onda_cut**2.0))

    #Constante de Atenuação

    B = gama0/(gamax*ux)

    #Par - S11 bruto

    s11_r = (np.tanh(gamax*L)+B*np.tanh(gama0*dL)-B*(1+B*np.tanh(gamax*L)*np.tanh(gama0*dL)))/(np.tanh(gamax*L)+B*np.tanh(gama0*dL)+B*(1+B*np.tanh(gamax*L)*np.tanh(gama0*dL)))

    #Coeficiente de ajuste de plano da porta 1

    R1 = np.exp(-gama0*L1)

    # coeficiente de ajuste

    s11_r_colocar = R1*R1*s11_r

    # Parâmetro S11 

    S11_r.append(abs(s11_r_colocar))

    

    
#Plot Coeficiente de Transmissao e Reflexão

#fig=plt.figure(num=1,figsize=(x,y))



plt.plot(F_grafic,T,'b-',label ='T_Calc',alpha=0.5,markevery=skip_point)

plt.plot(F_grafic,R,'r-',label ='$\Gamma$_Calc',alpha=0.5,markevery=skip_point)



plt.xlim(8.2,12.4)

#plt.ylim(0,1.2)

plt.xlabel("Freq(GHz)",fontsize = fonte)

plt.ylabel("T(a.u)",fontsize = fonte)

plt.title("Transmission and Reflection Coeficiente", fontsize = fonte)

plt.legend(fontsize = fonte).get_frame().set_facecolor('0.95')

#plt.savefig(u'Grafico_8.jpg')

plt.show()

#plot Impedância Real

#fig=plt.figure(num=1,figsize=(x,y))

plt.plot(F_grafic,Z_nrw,'-',label="Impedância real")

plt.xlim(8.2,12.4)

plt.xlabel("Freq(GHz)")

plt.title("Impedância Real",fontsize = fonte)

plt.ylabel("Z($\Omega$) - normalizada com do Vácuo")

plt.legend(fontsize = fonte).get_frame().set_facecolor('0.95')

#plt.savefig(u'Grafico_1x.jpg')

plt.show()
#plot e

#fig=plt.figure(num=1,figsize=(x,y))



plt.plot(F_grafic,er_r,'-',label="$\epsilon_{r}$'_Calc",alpha=0.5,markevery=skip_point)

plt.plot(F_grafic,er_i,'-',label='$\epsilon_{r}$"_Calc',alpha=0.5,markevery=skip_point)

plt.xlim(8.2,12.4)

plt.xlabel("Freq(GHz)",fontsize = fonte)

plt.ylabel('$\epsilon_{r}$"'+",$\epsilon_{r}$' (a.u)",fontsize = fonte)

plt.title("Permissivity",fontsize = fonte)

plt.legend(loc='best',fontsize = fonte).get_frame().set_facecolor('0.95')

#plt.savefig(u'Grafico_10.jpg')

plt.show()
#plot u

#fig=plt.figure(num=1,figsize=(x,y))



plt.plot(F_grafic,ur_r,'-',label = "$\mu_{r}$'_Calc",alpha=0.5,markevery=skip_point )

plt.plot(F_grafic,ur_i,'-',label ='$\mu_{r}$"_Calc',alpha=0.5,markevery=skip_point)

plt.xlim(8.2,12.4)

plt.xlabel("Freq(GHz)",fontsize = fonte)

plt.ylabel('$\mu_{r}$"'+",$\mu_{r}$' (a.u)",fontsize = fonte)

plt.title("Permeability",fontsize = fonte)

plt.legend(loc='best',fontsize = fonte).get_frame().set_facecolor('0.95')

#plt.savefig(u'Grafico_11.jpg')

plt.show()


#plot e and u modulo

#fig=plt.figure(num=1,figsize=(x,y))

plt.plot(F_grafic,ur_abs,'-',label="ur_abs")

plt.plot(F_grafic,er_abs,'-',label='er_abs')

plt.xlim(8.2,12.4)

plt.xlabel("Freq(GHz)",fontsize = fonte)

plt.ylabel("Permissivity and Permeability (a.u)",fontsize = fonte)

plt.title("Permeability and Permissivity (MÓDULO)",fontsize = fonte)

plt.legend(fontsize = fonte).get_frame().set_facecolor('0.95')

#plt.savefig(u'Grafico_12.jpg')

plt.show()
#plot refletividade Analítica



#fig=plt.figure(num=1,figsize=(x,y))

plt.plot(F_grafic,R_Zin,'-',label="Refletividade, método Zin")

plt.plot(F_grafic,S11_r,'r-',label="Refletividade, método livro NIST")



plt.xlim(8.2,12.4)

plt.xlabel("Freq(GHz)",fontsize = fonte)

plt.ylabel("S11 - Linear Mag(a.u)")

plt.title("Refletividade - Linear Mag",fontsize = fonte)

plt.legend(fontsize = fonte).get_frame().set_facecolor('0.95')

#plt.savefig(u'Grafico_12.jpg')

plt.ylim(0,1.1)

plt.show()




#---------------------Gravar Dados TXT---------------------



arqnew= open("./data_e_u.txt",'w')



arqnew.write("%8s,%4s,%4s,%4s,%4s\n"%('Fre(Hz)','er','ei','ur','ui'))



for n in range(0,len(F)):



    escrever = "%.4f,%.4f,%.4f,%.4f,%.4f\n"%(F[n],er_r[n],er_i[n],ur_r[n],ur_i[n])

    #escrever = "%.4f\n"%(F[n])



    arqnew.write(escrever)







arqnew.close()
