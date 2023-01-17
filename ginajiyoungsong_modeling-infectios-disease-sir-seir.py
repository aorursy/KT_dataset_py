# I need the parameter = Beta

# also Want to Know Ro:Reproductive number
import scipy.integrate as spi

import numpy as np

import pylab as pl



beta =  0.1934 # S ->I 감염율 = beta 를 구하지 못해서 논문에 나온 beta 값 참조 ( 2020.02.10 중국논문발표)

#우리나라도 중국과 마찬가지로 2차 지역감염이 시작되고 전염력이 2/18 당시 매우 높았던것으로 보여 논문의 beta 도입.



gamma =  1/14 # I ->R 회복율 = 평균 회복기간의 역수

t_inc = 1.0 

t_end = 150.0



# 2/18 31번확진자 나온날 기준으로 초기 SIR 모델 만듦.

S0  = 9772 ; I0  = 31;R0 = 2



N = S0 + I0 + R0 

S0  = 9772 /N  # susceptible hosts

I0  = 31 /N    # infectious hosts

R0 = 2 /N      # recovered hosts



Input = (S0, I0, 0.0)



Input
def simple_SIR(INT, t):

  '''The main set of equation'''

  Y=np.zeros((3))

  X = INT      #  S0,   I0 

  Y[0] = -beta * X[0] * X[1]

  Y[1] = beta*X[0]*X[1]  - gamma * X[1]

  Y[2] = gamma * X[1]

  return Y # for spicy.odeint



t_start =0.0 ; 

t_range = np.arange(t_start, t_end + t_inc, t_inc)

SIR= spi.odeint(simple_SIR, Input, t_range)



pl.figure(figsize=(15,8))

pl.plot(SIR[:, 0], '-g', label='Susceptibles')

pl.plot(SIR[:, 2], '-k', label='Recovereds')

pl.plot(SIR[:, 1], '-r', label='Infectious')

pl.legend(loc=0)

pl.title('Prediction of Simple nCOV-19 SIR model')

pl.xlabel('Time(day)')

pl.ylabel('individuals')

pl.show()
beta = 0.1934 # S ->I 감염율

gamma= 1/14 # I ->R 회복율 = 회복기간의 역수



nu = mu = 1/(70*365) # 자연사망율 반영



t_inc =1.0 ; t_end =150.0

'''

# Initial conditions

pop= 51780579 # 총인구수

test = 136707

test_ing = 28414

negative_tested = 102965 ; NT =negative_tested

print('positive_tested',test_ing + NT) # I0제외된 전체 test 받은 수



I0 = 5328 /test  # 3/3 기준 :확진자

S0 = ( test -I0 )/test # 3/3 기준 :양성판정+음성판정+검사중인 사람수( 총검사자 ) - 확진자'''



# 2/18 31번확진자 나온날 기준으로 초기 SIR 모델 만듦.

S0  = 9772 ; I0  = 31;R0 = 2



N = S0 + I0 + R0 

S0  = 9772 /N  # susceptible hosts

I0  = 31 /N    # infectious hosts

R0 = 2 /N      # recovered hosts

Input = (S0, I0, 0.0)



Input
def simple_SIR(INT, t):

  '''The main set of equation'''

  Y=np.zeros((3))

  X = INT      #  S0,   I0 

  Y[0] = -beta * X[0] * X[1] - mu * X[0]

  Y[1] = beta*X[0]*X[1]  - gamma * X[1] - mu * X[1]

  Y[2] = gamma * X[1] - mu * X[2]                   # 자연사망자 제외 (위와 식이 조금 변형됨)

  return Y # for spicy.odeint



t_start =0.0 ; 

t_range = np.arange(t_start, t_end + t_inc, t_inc)

SIR= spi.odeint(simple_SIR, Input, t_range)



pl.figure(figsize=(15,8))

pl.plot(SIR[:, 0], '-g', label='Susceptibles')

pl.plot(SIR[:, 2], '-k', label='Recovereds')

pl.plot(SIR[:, 1], '-r', label='Infectious')

pl.legend(loc=0)

pl.title('Prediction of Simple nCOV-19 SIR model')

pl.xlabel('Time(day)')

pl.ylabel('individuals')

pl.show()
import scipy.integrate

import numpy

import matplotlib.pyplot as plt



# Initial conditions

# 인구수 51780579

S0 = 9772 # :양성판정+검사중+음성판정

E0 = 818  # :검사중

I0 = 31   #  :확진자

R0 = 2    #  :완치자 + 사망자(0)





# Time vector

t = numpy.linspace(0,100,100)



N = S0 + I0 + R0 # 모집단



S0_ = S0/N

E0 =  E0/N

I0 = I0/N

R0 = R0/N



print(S0_)# 양성판정 확진자 + 음성판정 격리해지자수 비율 proporion



'''

Ro = 0.5 # 1인당 전파율 1월20일 보고된 한국코로나바이러스 역학조사 논문에 나온 수치)

print('논문에 보고된 Ro 평균',Ro)

To = 336 # 14*24 회복기간 2주  *  24시간

beta = (Ro/To) + (Ro/(To*S0))   # Ro 이용해서 beta 구하는 논문수식

print('\n논문 수식으로 구한 감염율 beta =',beta) 

논문 수식으로 구한 감염율 beta = 0.0014882475196382277

1월 초기에 전염병예측모델과 현재 현황이 많이 달라져서 사용할수 없음. 그당시 Ro, beta  모두 작은값.

'''

beta = 0.1934 # 중국논문에 나온 beta 값

ramda = 1/14

sigma = 0.25





Input = (S0_, E0 , I0)



def SEIR(INT, t):

  '''The main set of equation'''

  Y=np.zeros((3))

  X = INT      #  S0,   I0 

  Y[0] = mu -beta * X[0] * X[2] - mu *X[0]

  Y[1] = beta*X[0]*X[2]  - sigma * X[1] - mu * X[1]

  Y[2] = sigma * X[1] - gamma * X[2] - mu * X[2] #(자연사망자 제외)

  return Y # for spicy.odeint

  

  

t_start =0.0 ; t_end = 150 ; t_inc = 1.0

t_range = np.arange(t_start, t_end + t_inc, t_inc)

SEIR= spi.odeint(SEIR, Input, t_range)



Rec =1. - (SEIR[:,0]+ SEIR[:,1]+ SEIR[:,2])





pl.figure(figsize=(15,10))

pl.subplot(311)

pl.plot(SEIR[:,0], '-g', label='Susceptibles');pl.legend(loc=0)

pl.title('Prediction of nCOV-19 SEIR model')

pl.xlabel('Time(days)') # 국내 전염병의 추세가 하루하루 다르기 때문에.. 일주일단위보다는 1일단위로 보는게 맞는듯..

pl.ylabel('Susceptibles')



pl.subplot(312)

pl.plot(SEIR[:,1], '-b', label='Exposed')

pl.plot(SEIR[:,2], '-r', label='Infectious');pl.legend(loc=0)

pl.xlabel('Time(days)')

pl.ylabel('Infectious')



pl.subplot(313)

pl.plot(Rec, '-k', label='Recovereds')

pl.xlabel('Time(days)')

pl.legend(loc=0)

pl.ylabel('Recovereds')

pl.show()





# 2/18 기준 확진자수로 그렸던 SIR 모델과 모습이 유사. 그당시 20일 지난 시점으로 보여짐.

#  3/3 기준 한국내 전염병 전파되서 확진자가 5천명을 넘음





# Initial conditions

# pop= 51780579 # 총인구수

test = 136707

test_ing = 28414

negative_tested = 102965 ; NT =negative_tested

print('positive_tested',test_ing + NT) # I0제외된 전체 test 받은 수





S0 = test 

E0 = test_ing

I0 = 5328

R0 = 41





# Time vector

t = numpy.linspace(0,100,100)



N = S0 + I0 + R0 # 모집단



S0_ = S0/N

I0 = I0/N

R0 = R0/N



beta= 0.1934



ramda = 1/14

gamma = 0.25



print('중국논문 beta ', beta)  







# ODEs 전염병예측모델에서 가장 전통적인 SIR 모델을 정의함

def SIR_model(y, t, beta, ramma):

    S, I, R = y

    

    dS_dt = -beta*S*I

    dI_dt = beta*S*I - ramma*I

    dR_dt = ramma*I

    

    return([dS_dt,dI_dt,dR_dt])

# Result

solution = scipy.integrate.odeint(SIR_model, [S0_,I0,R0], t, args=(beta, ramda))

solution = numpy.array(solution)



# plot result

plt.figure(figsize=(10,6))

plt.plot(t, solution[:, 0], label="S(t)")

plt.plot(t, solution[:, 1], label="I(t)")

plt.plot(t, solution[:, 2], label="R(t)")

plt.grid()

plt.legend()

plt.xlabel("Time")

plt.ylabel("Proportion")

plt.title("SIR model")

plt.box(False)

plt.show()