x = 1

type(x)
2 ** 200
5 / 2
5 // 2
x = 0.000005

y = 5e-6

print(x == y)
x = 1400000.00

y = 1.4e6

print(x == y)
float(1)
0.1 + 0.2 == 0.3
print("0.1 = {0:.17f}".format(0.1))

print("0.2 = {0:.17f}".format(0.2))

print("0.3 = {0:.17f}".format(0.3))
complex(1, 2)
1 + 2j
c = 3 + 4j
c.real  # real part
c.imag  # imaginary part
c.conjugate()  # complex conjugate
abs(c)  # magnitude, i.e. sqrt(c.real ** 2 + c.imag ** 2)
message = "what do you like?"

response = 'spam'
# length of string

len(response)
# Make upper-case. See also str.lower()

response.upper()
# Capitalize. See also str.title()

message.capitalize()
# concatenation with +

message + response
# multiplication is multiple concatenation

5 * response
# Access individual characters (zero-based indexing)

message[0]
type(None)
return_value = print('abc')
print(return_value)
result = (4 < 5)

result
type(result)
print(True, False)
bool(2014)
bool(0)
bool(3.1415)
bool(None)
bool("")
bool("abc")
bool([1, 2, 3])
bool([])