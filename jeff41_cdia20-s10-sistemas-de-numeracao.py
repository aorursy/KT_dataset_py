hex(25)
bin(35)
int("0x1C", 16)
int("0b10101", 2)
int(bin(1000), 2)
int(hex(1000), 16)
# binário
print( bin(87) )

# hexadecimal
print( hex(87) )
# binário
print( bin(195) )

# hexadecimal
print( hex(195) )
# binário
print( bin(18) )

# hexadecimal
print( hex(18) )
# binário
print( bin(119) )

# hexadecimal
print( hex(119) )
# binário
print( bin(93) )

# hexadecimal
print( hex(93) )
# binário
print( bin(234) )

# hexadecimal
print( hex(234) )
# decimal
print( int("0b10010011", 2) )

# hexadecimal
print( hex(int("0b10010011", 2)) )
# decimal
print( int("0b00101101", 2) )

# hexadecimal
print( hex(int("0b00101101", 2)) )
# decimal
print( int("0b01001011", 2) )

# hexadecimal
print( hex(int("0b01001011", 2)) )
# decimal
print( int("0b10011110", 2) )

# hexadecimal
print( hex(int("0b10011110", 2)) )
# decimal
print( int("0b01011100", 2) )

# hexadecimal
print( hex(int("0b01011100", 2)) )
# decimal
print( int("0b11000001", 2) )

# hexadecimal
print( hex(int("0b11000001", 2)) )
# binário
print( bin(int("0x7D", 16)) )

# decimal
print( int("0x7D", 16) )
# binário
print( bin(int("0xA1", 16)) )

# decimal
print( int("0xA1", 16) )
# binário
print( bin(int("0x59", 16)) )

# decimal
print( int("0x59", 16) )
# binário
print( bin(int("0xBC", 16)) )

# decimal
print( int("0xBC", 16) )
# binário
print( bin(int("0x96", 16)) )

# decimal
print( int("0x96", 16) )
# binário
print( bin(int("0x04", 16)) )

# decimal
print( int("0x04", 16) )
# Em um arquivo binhex.py

def main():
    n = 100
    # separa cada coluna com uma tabulacao: \t
    print("Dec.\tHex.\tBin.")
    for i in range(n+1):
        print("{}\t{}\t{}".format(i,hex(i),bin(i)))
        
main()
import math

def dec2bin(n):
    q = math.floor(n / 2)
    r = n % 2
    nb = str(r)
    
    while (q > 0):
        r = q % 2
        q = math.floor(q / 2)
        nb = str(r) + nb
        
    return "0b" + nb
        
def main():
    n = int( input("Entre com um inteiro positivo: ") )
    print( dec2bin(n) )
    
main()