from numpy import *

from scipy.interpolate import *

from matplotlib.pyplot import *
x = array((1,2,6,8,9,10,11,13,14,15,16,17,18,19,20,21))

y = array((0,2,4,6,19,27,34,69,96,117,134,172,227,309,369,450))
p3 = polyfit(x,y,5)
print("Polynomial regression orde 5 y = {:.4f} + {:.4f}x + {:.4f}x^2 + {:.4f}x^3 + {:.4f}x^4 + {:.4f}x^5".format(p3[0],p3[1],p3[2],p3[3],p3[4],p3[5]))
ax = plot(x,y,'ro')

plot(x,polyval(p3,x),'b')

title('Kurva Positif Covid-19 di Indonesia')

grid()

xlabel('Tanggal')

ylabel('Kasus Positif Covid-19')
z = array((1,2,6,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25))
ax = plot(x,y,'ro')

plot(z,polyval(p3,z),'b')

title('Polynomial Regression Covid-19 Indonesia')

grid()

xlabel('Tanggal')

ylabel('Kasus Positif Covid-19')