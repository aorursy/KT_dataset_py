import random 

table=list([[" "," "," "],[" "," "," "],[" "," "," "]])

done=[];turn=0;person=0;pc=0;cnt=0;winner=5;flage=0

pc=random.randint(0, 1)

if(pc==1):

    turn=1;person=0

    print("Computer is first player and computer use 1")

    print("You are second player and your sign is 0")

else:

    turn=0;person=1

    print("You are first player and your sign is 1")

    print("Computer is second player and computer use 0")

print("Starting index is 1 ")

while(1):

    print("|---|---|---|")

    print(f"| {table[0][0]} | {table[0][1]} | {table[0][2]} |")

    print("|---|---|---|")

    print(f"| {table[1][0]} | {table[1][1]} | {table[1][2]} |")

    print("|---|---|---|")

    print(f"| {table[2][0]} | {table[2][1]} | {table[2][2]} |")

    print("|---|---|---|")

    print(" ")

    while(1):

        if(turn==1):

            r=random.randint(0, 2)

            c=random.randint(0, 2)

            r+=1;c+=1

            value=pc

        else:

            print("Enter row position ")

            r=int(input(""))

            print("Enter colum position ")

            c=int(input(""));value=person

        p=(r)*10+(c)

        if(p not in done):

                done.append(p)

                table[r-1][c-1]=value

                break

    turn = 0 if turn==1 else 1

    size=len(done)

    if(size==9):

        print("Game Over");flage=1

    cnt+=1

    if(cnt>2):

        if((table[0][0]==table[0][1]==table[0][2]) or (table[0][0]==table[1][0]==table[2][0])):

            winner=table[0][0]

        elif((table[2][2]==table[1][2]==table[0][2]) or (table[2][0]==table[2][1]==table[2][2])):

            winner=table[2][2]

        elif((table[1][1]==table[0][1]==table[2][1]) or (table[1][1]==table[1][0]==table[1][2])):

            winner=table[1][1]

        elif((table[1][1]==table[0][0]==table[2][2]) or (table[1][1]==table[0][2]==table[2][0])):

            winner=table[1][1]

        if(winner==pc):

            print("Computer Win");flage=1

        if(winner==person):

            print("You win");flage=1

        if(flage==1):

            print("|---|---|---|")

            print(f"| {table[0][0]} | {table[0][1]} | {table[0][2]} |")

            print("|---|---|---|")

            print(f"| {table[1][0]} | {table[1][1]} | {table[1][2]} |")

            print("|---|---|---|")

            print(f"| {table[2][0]} | {table[2][1]} | {table[2][2]} |")

            print("|---|---|---|")

            print(" ")

            break