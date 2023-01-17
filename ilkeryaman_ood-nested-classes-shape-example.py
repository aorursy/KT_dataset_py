class Shape:
    char = "*"
    separator = "-"

    @staticmethod
    def draw_separator():
        print(Shape.separator * 50)

    class Line:
        def __init__(self, length):
            self.length = length

        def draw(self):
            print(*self.length * "*")

    class Rectangle:
        def __init__(self, a, b):
            self.a, self.b = a, b

        def draw(self):
            print(*Shape.char * self.a)
            for x in range(1, self.b - 1):
                print(Shape.char, " " * ((self.a * 2)-3), Shape.char, sep="")
            print(*Shape.char * self.a)

    class Square:
        def __init__(self, a):
            self.a = a

        def draw(self):
            print(*Shape.char * self.a)
            for x in range(1, self.a - 1):
                print(Shape.char, " " * ((self.a * 2)-3), Shape.char, sep="")
            print(*Shape.char * self.a)

    class Triangle:
        def __init__(self, ceil):
            self.ceil = ceil

        def draw(self):
            print(Shape.char)
            print(*Shape.char * 2)
            for x in range(2, self.ceil):
                print(*Shape.char, " " * (x - 2), Shape.char)
            print(* Shape.char * ((self.ceil // 2) + 2))
shape_list = [Shape.Line(7), Shape.Rectangle(18, 6), Shape.Square(7), Shape.Triangle(9)]
for shape in shape_list:
    shape.draw()
    Shape.draw_separator()  # Attention! Usage of static method!