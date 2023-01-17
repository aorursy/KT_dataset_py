def encrypt(text,key):

    b=""

    for i in text:

        if i.isalpha()==True:

            c=ord(i)

            d=c+key

            if d<=(122):

                b=b+chr(d) 

            elif d>122:

                a=122-c

                x=key-a

                b=b+chr(96+x)

        else:

            b=b+i

    return b

    

encrypt("hello world!",3)
def decrypt(text,key):

    b=""

    for i in text:

        if i.isalpha()==True:

            c=ord(i)

            d=c-key

            if d<=97: 

                j=c-97

                g=key-j

                t=123-g

                b=b+chr(t)

            elif d<=(122):               

                    b=b+chr(d)           

        else:

            b=b+i

    return b

    

decrypt("khoor zruog!",3) 