from math import sqrt

class Punto:

    __x = -1

    __y = 0

    def __init__(self, x = 0, y = 0):

        self.__x = x

        self.__y = y

    def __str__(self):

        return '(x,y)=({}, {})'.format(self.__x, self.__y)

    def setX(self, v):

        self.__x = v

    def getX(self):

        return self.__x

    def setY(self, v):

        self.__y = v

    def getY(self):

        return self.__y

    def mostrarDatos(self):

        print(self)

    def mover(self, x, y):

        self.setX(x)

        self.setY(y)

    def distanciaEuclidea(self, otroPunto):

        distancia = sqrt((otroPunto.__x-self.__x)**2+(otroPunto.__y-self.__y)**2)

        return distancia
class Lista:

    __puntos = []

    def __init__(self):

        self.__puntos=[]

    def agregarPunto(self, unPunto):

        self.__puntos.append(unPunto)

    def mostrarPuntos(self):

        print('Imprimiendo puntos de la colección')

        for punto in self.__puntos:

            punto.mostrarDatos()

    def testListaPuntos(self):

        p1 = Punto(4, 5)

        p2 = Punto(3, 4)

        p3 = Punto(-9, 5)

        p4 = Punto(3, 2)

        p5 = Punto(1, 7)

        self.agregarPunto(p1)

        self.agregarPunto(p2)

        self.agregarPunto(p3)

        self.agregarPunto(p4)

        self.agregarPunto(p5)

    def getUnPunto(self, indice):

        return self.__puntos[indice]

    def calcularDistanciasP0(self, unPunto):

        for i in range(len(self.__puntos)):

            distancia = unPunto.distanciaEuclidea(self.__puntos[i])

            print('Distancia del punto {} al punto {} es {}'.format(unPunto, self.__puntos[i], distancia))
#Programa Principal

listaPuntos = Lista()

listaPuntos.testListaPuntos()

punto0 = listaPuntos.getUnPunto(0)

listaPuntos.calcularDistanciasP0(punto0)

listaPuntos.mostrarPuntos()
unEntero=5

listaPuntos.agregarPunto(unEntero)
punto0 = listaPuntos.getUnPunto(0)

listaPuntos.calcularDistanciasP0(punto0)

listaPuntos.mostrarPuntos()