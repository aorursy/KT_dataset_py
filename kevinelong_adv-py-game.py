print(-1 % 4)
print(0 % 4)
print(1 % 4)
print(2 % 4)
print(3 % 4)
print(4 % 4)
print(5 % 4)


# ENUM ENUMERATION IN PYTHON
class CardinalDirection:
    NORTH = 0  # UP
    EAST = 1  # RIGHT
    SOUTH = 2  # DOWN
    WEST = 3  # LEFT


class TurnDirection:
    LEFT = 0
    RIGHT = 1


class Player:
    # CONSTRUCTOR
    def __init__(self):
        self.facing = CardinalDirection.EAST

    def turn(self, direction):
        # USE MODULO DIVISION OPERATOR TO WRAP AROUND
        if direction == TurnDirection.LEFT:
            self.facing = (self.facing - 1) % 4
        elif direction == TurnDirection.RIGHT:
            self.facing = (self.facing + 1) % 4

    def show(self):
        # A SWITCH STATEMENT IN PYTHON IS JUST A LOT OF ELIFS
        if self.facing == 0:
            symbol = "^"
        elif self.facing == 1:
            symbol = ">"
        elif self.facing == 2:
            symbol = "v"
        elif self.facing == 3:
            symbol = "<"
        else:
            raise ValueError
        print(symbol)


p = Player()

p.show()

p.turn(TurnDirection.RIGHT)
p.show()
p.turn(TurnDirection.RIGHT)
p.show()
p.turn(TurnDirection.RIGHT)
p.show()
p.turn(TurnDirection.RIGHT)
p.show()
text_symbols = "North\tEast\tSouth\tWest"
print(text_symbols)

symbols = text_symbols.split("\t") # TEAR THEM APART AT THE SEPERATOR

# for s in symbols:
#     print(s)
print("\n".join(symbols))
print("--".join(symbols)) #GLUE THEM BACK TOGETHER

parts = "192.168.1.1".split(".")
print(parts[1])

parts = "/foo/bar/".split("/")

print(f"0={parts[0]}")
# /
print(f"1={parts[1]}")
# /
print(f"2={parts[2]}")
# /
print(f"3={parts[3]}")

print(f"4={parts[4]}")


