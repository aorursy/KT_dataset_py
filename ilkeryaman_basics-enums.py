from enum import Enum, IntEnum, unique, auto
class Color(Enum):
    GREEN = 1
    RED = 2
    YELLOW = 3
    LEMONADE = 3
@unique
class HttpStatus(IntEnum):
    OK = 200
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
class Car(Enum):
    MERCEDES
    BMW
    AUDI
    SKODA
class HeaderParams(Enum):
    CONTENT_TYPE = auto()
    ACCEPT = auto()
    EXECUTION_ID = auto()
print(Color.RED)
print(repr(Color.RED))
print(Color.RED.name)
print(Color.RED.value)
print(Color.RED.value == 2)
print(list(Color))            # Attention please!
print(list(HttpStatus))
print(list(HeaderParams))
print(HttpStatus.OK)
print(HttpStatus.OK.value)