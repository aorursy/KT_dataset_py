def traduzir(codigo):

    

    string = ""

    

    chave = "'abcdefghijklmnopqrstuvwxyz"

    

    for numero in codigo:

        if (numero >= 0 and numero <= 26):

            string += chave[numero]

    

    #ou

    '''

    for numero in codigo:

        if (numero >= 0 and numero <= 26): 

            if numero == 39:

                string += "'"

            else:

                string += chr(numero + 96) 

                

    '''

                

    return string
codigo = [0,1,9,12,20,15,14,0,27]



print(traduzir(codigo))
ord("a")