from math import sqrt

def simple_LR(X,Y):

  # no of Training Samples
  l = len(X)
  # sum x, sum x_squared and Mean of x
  sum_x = 0;
  sum_x2 = 0
  for x in X:
    sum_x += x
    sum_x2 += x**2
  x_mean = sum_x / l

  # sum y, sum y_squared and Mean of y
  sum_y = 0;
  sum_y2 = 0
  for y in Y:
    sum_y += y
    sum_y2 += y**2
  y_mean = sum_y / l

  # Variability Sxx, Syy and Sxy
  # Sxx
  Sxx = 0
  for x in X:
    Sxx += (x - x_mean)**2
  # Syy
  Syy = 0
  for y in Y:
    Syy += (y - y_mean)**2
  #Sxy
  Sxy = 0
  for i in range(l):
    Sxy += (X[i] - x_mean)*(Y[i] - y_mean)

  # Sum of the Squares: SST,SSR,SSE
  # SST
  SST = Syy
  # SSR
  SSR = (Sxy**2)/Sxx
  # SSE
  SSE = SST - SSR

  # Evaluation of Regression Coefficients (MODEL : y = b0 + b1x)
  # b1
  b1 = Sxy / Sxx
  # b0
  b0 = y_mean - b1*x_mean

  # STATISTICS Section

  # Standard Error "s"
  s = sqrt(SSE/(l-2))

  # Standard Error in Intercept Estimate "s_b0"
  s_b0 = s*sqrt((sum_x2)/(l*Sxx))

  # Standard Error in Slope Estimate "s_b1"
  s_b1 = s*sqrt(1/Sxx)

  # t-Scores
  t_b0 = b0 / s_b0
  t_b1 = b1 / s_b1

  # Coefficient of determination
  r2 = SSR/SST
  r2_adj = 1-(((1-r2)*(l-1))/(l-2))

  print('*'*124)
  print("Univariate Linear Regression model : y = " , b0 , " + " , b1 , "x")
  print('*'*124)
  print("STATISTICS")
  print()
  print("Sxx :",Sxx)
  print("Syy :",Syy)
  print("Sxy :",Sxy)
  print(" ")
  print("SSE :",SSE)
  print("SSR :",SSR)
  print("SST :",SST)
  print(" ")
  print("Standard Error s :",s)
  print("Standard Error in coefficients: s-b0 = ",s_b0," s-b1 = ",s_b1)
  print(" ")
  print("Student-t tests:")
  print("t(b0) = ",t_b0)
  print("t(b1) = ",t_b1)
  print(" ")
  print("Coefficient of Determination :")
  print("r-squared r^2 = ",r2)
  print("Adjusted r-squared r^2(adj) = ",r2_adj)
  print('*'*124)
  
X = [6,16,9,8,14,11,12,10,18,5,26,8,8,9,5]
Y = [76,10,44,47,23,19,13,19,8,44,4,31,24,59,37]
simple_LR(X,Y)