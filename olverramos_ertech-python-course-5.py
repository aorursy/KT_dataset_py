class Fraccion:
    numerador = None
    denominador = None
f = Fraccion()
f.numerador = 3
f.denominador = 4

print (f.numerador)

print (f.denominador)
print (f)
print (type(f))
class Fraccion:
    numerador = None
    denominador = None
    
    def get_decimal(self):
        if self.numerador is not None and self.denominador is not None and self.denominador > 0:
            return self.numerador / self.denominador
        return None
    
f = Fraccion()
f.numerador = 3
f.denominador = 4

print (f.get_decimal())
class Fraccion:
    numerador = None
    denominador = None
    
    def __init__(self, numerador=0, denominador=1):
        self.numerador = numerador
        self.denominador = denominador
        if self.denominador < 0:
            self.numerador = self.numerador * -1
            self.denominador = self.denominador * -1
    
    def get_decimal(self):
        if self.numerador is not None and self.denominador is not None and self.denominador > 0:
            return self.numerador / self.denominador
        return None
    
f1 = Fraccion(3, 4)

print (f1.get_decimal())

print (f1)

f2 = Fraccion(3)

print (f2.get_decimal())

f3 = Fraccion()

print (f3.get_decimal())


class Fraccion:
    numerador = None
    denominador = None
    
    def __init__(self, numerador=0, denominador=1):
        self.numerador = numerador
        self.denominador = denominador
        if self.denominador < 0:
            self.numerador = self.numerador * -1
            self.denominador = self.denominador * -1
    
    def __str__(self):
        if self.denominador != 1:
            return f"{self.numerador}/{self.denominador}"
        return f"{self.numerador}"
    
    def get_decimal(self):
        if self.numerador is not None and self.denominador is not None and self.denominador > 0:
            return self.numerador / self.denominador
        return None
f1 = Fraccion(3, 4)

print (f1)

print (f"Las {f1} partes del día estuve despierto")

f2 = Fraccion(3)

print (f2)

f3 = Fraccion()

print (f3)

f1 + f2
# Definimos el 
def MCD(m,n):
    while m % n != 0:
        mViejo = m
        nViejo = n

        m = nViejo
        n = mViejo%nViejo
    return n
class Fraccion:
    numerador = None
    denominador = None
    
    def __init__(self, numerador=0, denominador=1):
        self.numerador = numerador
        self.denominador = denominador
        if self.denominador < 0:
            self.numerador = self.numerador * -1
            self.denominador = self.denominador * -1
    
    def __str__(self):
        if self.denominador != 1:
            return f"{self.numerador}/{self.denominador}"
        return f"{self.numerador}"
    
    def get_decimal(self):
        if self.numerador is not None and self.denominador is not None and self.denominador > 0:
            return self.numerador / self.denominador
        return None
    
    def simplificar(self):
        comun = MCD(self.numerador, self.denominador)
        self.numerador = int(self.numerador / comun)
        self.denominador = int(self.denominador / comun)
    
    def __add__(self, f):
        result = Fraccion()
        result.denominador = self.denominador * f.denominador
        result.numerador = self.numerador * f.denominador + f.numerador * self.denominador 
        result.simplificar()
        return result
    
f1 = Fraccion(3, 4)
f2 = Fraccion(7, 8)

print (f1 + f2)

class Fraccion:
    numerador = None
    denominador = None
    
    def __init__(self, numerador=0, denominador=1):
        self.numerador = numerador
        self.denominador = denominador
        if self.denominador < 0:
            self.numerador = self.numerador * -1
            self.denominador = self.denominador * -1
            
    
    def __str__(self):
        if self.denominador != 1:
            return f"{self.numerador}/{self.denominador}"
        return f"{self.numerador}"
    
    def get_decimal(self):
        if self.numerador is not None and self.denominador is not None and self.denominador > 0:
            return self.numerador / self.denominador
        return None
    
    def simplificar(self):
        comun = MCD(self.numerador, self.denominador)
        self.numerador = int(self.numerador / comun)
        self.denominador = int(self.denominador / comun)
    
    def __add__(self, f):
        result = Fraccion()
        result.denominador = self.denominador * f.denominador
        result.numerador = self.numerador * f.denominador + f.numerador * self.denominador 
        result.simplificar()
        return result
    
    def __eq__(self, f):
        return (self.numerador * f.denominador) == (self.denominador * f.numerador)
f1 = Fraccion(3, 4)
f2 = Fraccion(7, 8)
f3 = Fraccion(9, 12)

print(f1 == f2)
print(f1 == f3)
# R/
class Fraccion:
    numerador = None
    denominador = None
    
    def __init__(self, numerador=0, denominador=1):
        self.numerador = numerador
        self.denominador = denominador
        if self.denominador < 0:
            self.numerador = self.numerador * -1
            self.denominador = self.denominador * -1
                
    def __str__(self):
        if self.denominador != 1:
            return f"{self.numerador}/{self.denominador}"
        return f"{self.numerador}"
    
    def get_decimal(self):
        if self.numerador is not None and self.denominador is not None and self.denominador > 0:
            return self.numerador / self.denominador
        return None
    
    def simplificar(self):
        comun = MCD(self.numerador, self.denominador)
        self.numerador = int(self.numerador / comun)
        self.denominador = int(self.denominador / comun)
    
    def __add__(self, f):
        result = Fraccion()
        result.denominador = self.denominador * f.denominador
        result.numerador = (self.numerador * f.denominador) + (f.numerador * self.denominador)
        result.simplificar()
        return result
    
    def __eq__(self, f):
        return (self.numerador * f.denominador) == (self.denominador * f.numerador)
    
    def __neg__(self):
        result = Fraccion(-1 * self.numerador, self.denominador)
        return result
    
    def __invert__(self):
        result = Fraccion(self.denominador, self.numerador)
        return result
    
    def __sub__(self, f):
        return self + (-f)
    
    def __mul__(self, f):
        result = Fraccion()
        result.denominador = self.denominador * f.denominador
        result.numerador = self.numerador * f.numerador
        result.simplificar()
        return result
    
    def __div__(self, f):
        result = Fraccion()
        result.numerador = self.numerador * f.denominador
        result.denominador = self.denominador * f.numerador
        result.simplificar()
        return result
    
    def __pow__(self, exponente):
        result = Fraccion()
        result.numerador = self.numerador ** exponente
        result.denominador = self.denominador ** exponente
        result.simplificar()
        return result
    
# R/ 
class Fraccion:
    numerador = None
    denominador = None
    
    def __init__(self, numerador=0, denominador=1):
        self.numerador = numerador
        self.denominador = denominador
        if self.denominador < 0:
            self.numerador = self.numerador * -1
            self.denominador = self.denominador * -1
                
    def __str__(self):
        if self.denominador != 1:
            return f"{self.numerador}/{self.denominador}"
        return f"{self.numerador}"
    
    def get_decimal(self):
        if self.numerador is not None and self.denominador is not None and self.denominador > 0:
            return self.numerador / self.denominador
        return None
    
    def simplificar(self):
        comun = MCD(self.numerador, self.denominador)
        self.numerador = int(self.numerador / comun)
        self.denominador = int(self.denominador / comun)
    
    def __add__(self, f):
        result = Fraccion()
        result.denominador = self.denominador * f.denominador
        result.numerador = (self.numerador * f.denominador) + (f.numerador * self.denominador)
        result.simplificar()
        return result
    
    def __eq__(self, f):
        return (self.numerador * f.denominador) == (self.denominador * f.numerador)
    
    def __neg__(self):
        result = Fraccion(-1 * self.numerador, self.denominador)
        return result
    
    def __invert__(self):
        result = Fraccion(self.denominador, self.numerador)
        return result
    
    def __sub__(self, f):
        return self + (-f)
    
    def __mul__(self, f):
        result = Fraccion()
        result.denominador = self.denominador * f.denominador
        result.numerador = self.numerador * f.numerador
        result.simplificar()
        return result
    
    def __div__(self, f):
        result = Fraccion()
        result.numerador = self.numerador * f.denominador
        result.denominador = self.denominador * f.numerador
        result.simplificar()
        return result
    
    def __pow__(self, exponente):
        result = Fraccion()
        result.numerador = self.numerador ** exponente
        result.denominador = self.denominador ** exponente
        result.simplificar()
        return result
    
    def __lt__(self, f):
        return (self.numerador * f.denominador) < (self.denominador * f.numerador)
        
    def __le__(self, f):
        return (self.numerador * f.denominador) <= (self.denominador * f.numerador)
        
    def __gt__(self, f):
        return (self.numerador * f.denominador) > (self.denominador * f.numerador)
        
    def __ge__(self, f):
        return (self.numerador * f.denominador) >= (self.denominador * f.numerador)

    def __ne__(self, f):
        return (self.numerador * f.denominador) != (self.denominador * f.numerador)
                
f1 = Fraccion(3,4)
f2 = Fraccion(5,7)

f1 += f2
print (f1)
# R/