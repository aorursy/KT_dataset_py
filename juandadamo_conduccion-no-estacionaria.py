!pip install ht;
!pip install future;
!pip install fipy
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fipy import Variable, FaceVariable, CellVariable, Grid1D, ExplicitDiffusionTerm, TransientTerm, DiffusionTerm, Viewer
from fipy.tools import numerix
from scipy.special import erf # doctest: +SCIPY

### Geometría
nx = 50
Lx = 1.
dx = Lx/nx
#Definicion de una malla
mesh = Grid1D(Lx=Lx, dx = dx)
#acceso a los nodos de la malla
x = mesh.cellCenters[0]

#Definición de la variable solución del problema: T
T = CellVariable(name="solution variable",mesh=mesh, value=0.)

#Valor Inicial de la solución
T.setValue(0)

#Parámetro Físico de la ecuacion: Difusión a

a = 1.

# Condiciones de Borde Dirichlet 0 y 1
valueLeft = 1
valueRight = 0

# Aplicacion sobre la variable solucion del problema
T.constrain(valueRight, mesh.facesRight)
T.constrain(valueLeft, mesh.facesLeft)
# Definicion de la variable tiempo
timeStepDuration = 0.9 * dx**2 / (2 * a)
steps = 100
t_final = timeStepDuration * steps

# Definición de la ecuación a resolver: Ecuacion de Fourier
# Se trata de un termino transitorio y un termino difusivo
eqX = TransientTerm() == ExplicitDiffusionTerm(coeff=a)  

#Opcional para este problema introductorio, comparación con el resultado analítico
T_Analytical = CellVariable(name="analytical value", mesh=mesh)
T_Analytical.setValue(1 - erf(x / (2 * numerix.sqrt(a * t_final)))) 
# Preparación de la salida gráfica. Se visualizará la solución y la referencia analítica
viewer = Viewer(vars=(T,T_Analytical) )

# Resolución de la ecuación diferencial para cada paso temporal 
for step in range(steps):
    eqX.solve(var=T,dt=timeStepDuration)
    if __name__ == '__main__':
        viewer.plot('1Dtransient_%0d.png'%step)
nx = 50
Lx = 1.
dx = Lx/nx
mesh = Grid1D(Lx=Lx, dx = dx)

phi2 = CellVariable(name="solution variable",mesh=mesh)
phi2.setValue(0)
x = mesh.cellCenters[0]
time = Variable()
D = 1.

valueLeft = 1+ 0.5 * (1 + numerix.sin(.5*time))
fluxRight = -.5
phi2.constrain(valueLeft, mesh.facesLeft)
phi2.faceGrad.constrain([fluxRight], mesh.facesRight)
eqI = TransientTerm() == DiffusionTerm(coeff=D)
phiAnalytical = CellVariable(name="analytical value", mesh=mesh)
viewer2 = Viewer(vars=(phi2),datamin= 0,datamax=3);
dt = .5
while time() < 50:
    time.setValue(time() + dt)
    eqI.solve(var=phi2, dt=dt)
    if __name__ == '__main__':
        viewer2.plot()

