for fizzBuzz in range(1,100):
    if (fizzBuzz % 15 == 0):
        print("FizzBuzz")

    elif (fizzBuzz % 3 == 0):
        print("Fizz")

    elif (fizzBuzz % 5 == 0):
        print("Buzz")

    else:
        print(fizzBuzz)