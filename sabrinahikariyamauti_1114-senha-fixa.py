senha = 1                               # n1 op.

while senha != 2002:                    # n*

    senha = int(input())                # n2 op.
    if senha != 2002:                   # n1 op.
        print('Senha Invalida')         # n1 op.


    else:
        print('Acesso Permitido')       # n1 op.
        break                           # n1 op.


# total = 1  + n *   (2 + 1 + (1 + 1))   =   1 + n * (2 + 1 + 2)   =   1 + n*5   =   1 + 5n
#         |    |      |    --|--     --|--
#     Senha   While Senha   If       Else
