class Taxi:
    placa = None

class Persona:
    cedula = None
    nombre = None
    taxi = None # conduce y es de clase Taxi
    

class Persona:
    cedula = None
    nombre = None

class Taxi:
    placa = None
    conductor = None # es conducido por, es de clase Persona
    

class Persona:
    cedula = None
    nombre = None

class Taxi:
    placa = None
    conductores = None # es conducido por, es una lista de objetos de clase Persona
    

class Persona:
    cedula = None
    nombre = None

class Taxi:
    placa = None
    
class Persona_Taxi:    
    conductor = None # es conducido por, es de clase Persona
    taxi = None # conduce, es de clase Taxi
    
class Vehiculo:
    color = None
    placa = None
    
class Taxi(Vehiculo):
    registro_unico = None

