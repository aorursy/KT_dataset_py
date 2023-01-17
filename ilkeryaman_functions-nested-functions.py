def operate(a, b, operation):
    def addition():
        return a + b

    def subtraction():
        return a - b

    def multiplication():
        return a * b

    def division():
        return a / b

    if operation == "1":
        print(addition())
    elif operation == "2":
        print(subtraction())
    elif operation == "3":
        print(multiplication())
    elif operation == "4":
        print(division())
    else:
        print("Operation not found!")
operate(4, 2, "1")
operate(4, 2, "2")
operate(4, 2, "3")
operate(4, 2, "4")
operate(4, 2, "5")