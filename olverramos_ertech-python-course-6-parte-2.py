class Vehiculo:
    color = None
    placa = None
    
    def __init__(self, color=None, placa=None):
        self.color=color
        self.placa=placa
        
    def obtener_datos_principales(self):
        return f"El vehículo es de color {self.color} y su placa es {self.placa}"
                
class Carro(Vehiculo):
    cinturones = None
    puertas = None
    
    def __init__(self, color=None, placa=None, cinturones=None, puertas=None):
        self.color=color
        self.placa=placa
        self.cinturones=cinturones
        self.puertas=puertas
          
    def obtener_datos_carro(self):
        result = f"El vehículo es de color {self.color}, su placa es {self.placa}, "
        result += f"cuenta con {self.cinturones} cinturones y {self.puertas} puertas"
        return result
                
class Moto(Vehiculo):
    color_casco = None

    def __init__(self, color=None, placa=None, color_casco=None):
        self.color=color
        self.placa=placa
        self.color_casco=color_casco

    def obtener_datos_moto(self):
        result = f"El vehículo es de color {self.color}, su placa es {self.placa} y "
        result += f"cuenta con un casco de color {self.color_casco}"
        return result

c = Carro('Rojo', 'SSS237', 4, 4)
m = Moto('Azul', 'AR36F', 'Negro')
print (c.obtener_datos_principales())
print (c.obtener_datos_carro())
print (m.obtener_datos_moto())
class Vehiculo:
    color = None
    placa = None
    
    def __init__(self, color=None, placa=None):
        self.color=color
        self.placa=placa
        
    def obtener_datos_principales(self):
        return f"El vehículo es de color {self.color} y su placa es {self.placa}"
                
class Carro(Vehiculo):
    cinturones = None
    puertas = None
    
    def __init__(self, color=None, placa=None, cinturones=None, puertas=None):
        super().__init__(color, placa)
        self.cinturones=cinturones
        self.puertas=puertas
          
    def obtener_datos_carro(self):
        result = f"El vehículo es de color {self.color}, su placa es {self.placa}, "
        result += f"cuenta con {self.cinturones} cinturones y {self.puertas} puertas"
        return result
                
class Moto(Vehiculo):
    color_casco = None

    def __init__(self, color=None, placa=None, color_casco=None):
        super().__init__(color, placa)
        self.color_casco=color_casco

    def obtener_datos_moto(self):
        result = f"El vehículo es de color {self.color}, su placa es {self.placa} y "
        result += f"cuenta con un casco de color {self.color_casco}"
        return result

c = Carro('Rojo', 'SSS237', 4, 4)
m = Moto('Azul', 'AR36F', 'Negro')
print (c.obtener_datos_principales())
print (c.obtener_datos_carro())
print (m.obtener_datos_moto())
import json

class Vehiculo:
    color = None
    placa = None
    
    def __init__(self, color=None, placa=None):
        self.color=color
        self.placa=placa
        
    def obtener_datos_principales(self):
        return f"El vehículo es de color {self.color} y su placa es {self.placa}"
        
    def obtenerDict(self):
        data = {}
        data['color'] = self.color
        data['placa'] = self.placa
        return data
                
class Carro(Vehiculo):
    cinturones = None
    puertas = None
    
    def __init__(self, color=None, placa=None, cinturones=None, puertas=None):
        super().__init__(color, placa)
        self.cinturones=cinturones
        self.puertas=puertas
          
    def obtener_datos(self):
        result = super().obtener_datos_principales()
        result += f", cuenta con {self.cinturones} cinturones y {self.puertas} puertas"
        return result
    
    def obtenerJson(self):
        data = super().obtenerDict()
        data['cinturones'] = self.cinturones
        data['puertas'] = self.puertas
        return json.dumps(data)
                
class Moto(Vehiculo):
    color_casco = None

    def __init__(self, color=None, placa=None, color_casco=None):
        super().__init__(color, placa)
        self.color_casco=color_casco

    def obtener_datos(self):
        result = super().obtener_datos_principales()
        result += f" y cuenta con un casco de color {self.color_casco}"
        return result

    def obtenerJson(self):
        data = super().obtenerDict()
        data['color_casco'] = self.color_casco
        return json.dumps(data)
                
c = Carro('Rojo', 'SSS237', 4, 4)
m = Moto('Azul', 'AR36F', 'Negro')
print (c.obtener_datos_principales())
print (c.obtener_datos())
print (m.obtener_datos_principales())
print (m.obtener_datos())
print (c.obtenerJson())
print (m.obtenerJson())
print (c.obtenerDict())

class Vehiculo:
    __color = None
    __placa = None
    
    def __init__(self, color=None, placa=None):
        self.__color=color
        self.__placa=placa
        
    def obtener_datos_principales(self):
        return f"El vehículo es de color {self.__color} y su placa es {self.__placa}"
        
    def __obtenerDict(self):
        data = {}
        data['color'] = self.__color
        data['placa'] = self.__placa
        return data
                
    def obtener_color(self):
        return self.__color
                
    def obtener_placa(self):
        return self.__placa
    
    def cambiar_color(self, color):
        self.__color = color
    
    def cambiar_placa(self, placa):
        self.__placa = placa
    
class Carro(Vehiculo):
    __cinturones = None
    puertas = None
    
    def __init__(self, color=None, placa=None, cinturones=None, puertas=None):
        super().__init__(color, placa)
        self.__cinturones=cinturones
        self.puertas=puertas
          
    def obtener_datos(self):
        result = super().obtener_datos_principales()
        result += f", cuenta con {self.__cinturones} cinturones y {self.puertas} puertas"
        return result
    
    def obtenerJson(self):
        data = super().__obtenerDict()
        data['cinturones'] = self.__cinturones
        data['puertas'] = self.puertas
        return json.dumps(data)
                          
    def obtener_cinturones(self):
        return self.__cinturones
    
    def cambiar_cinturones(self, cinturones):
        self.__cinturones = cinturones
    
class Moto(Vehiculo):
    color_casco = None

    def __init__(self, color=None, placa=None, color_casco=None):
        super().__init__(color, placa)
        self.color_casco=color_casco

    def obtener_datos(self):
        result = super().obtener_datos_principales()
        result += f" y cuenta con un casco de color {self.color_casco}"
        return result

    def obtenerJson(self):
        data = super().__obtenerDict()
        data['color_casco'] = self.color_casco
        return json.dumps(data)
                
c = Carro('Rojo', 'SSS237', 4, 4)
m = Moto('Azul', 'AR36F', 'Negro')
print (c.obtener_datos_principales())
print (c.obtener_datos())
print (m.obtener_datos_principales())
print (m.obtener_datos())
m.color_casco
print(m.__color)
m.obtener_color()
m.cambiar_color('Negro')
m.obtener_color()
c.__obtenerDict()