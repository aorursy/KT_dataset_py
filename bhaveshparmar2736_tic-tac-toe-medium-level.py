import random 

def hvalue(r,c):

    if(table[r][c]==" "):

        return 0

    elif table[r][c]==aposit:

        return 1

    else:

        return 3

def print_table():

    print("|---|---|---|")

    print(f"| {table[0][0]} | {table[0][1]} | {table[0][2]} |")

    print("|---|---|---|")

    print(f"| {table[1][0]} | {table[1][1]} | {table[1][2]} |")

    print("|---|---|---|")

    print(f"| {table[2][0]} | {table[2][1]} | {table[2][2]} |")

    print("|---|---|---|")

    print(" ")

    

table=list([[" "," "," "],[" "," "," "],[" "," "," "]])

rest=[11,12,13,21,22,23,31,32,33]

done=[];turn=0;person=0;pc=0;cnt=0;winner=5;flage=0;aposit=0;best=0;r_best=0;c_best=0;h=0

rh=0;lh=0;uh=0;dh=0;d1=0;d2=0;d3=0;d4=0

pc=random.randint(0, 1)

if(pc==1):

    turn=1;person=0

    print("Computer is first player and computer use 1")

    print("You are second player and your sign is 0")

else:

    turn=0;person=1

    print("You are first player and your sign is 1")

    print("Computer is second player and computer use 0")

aposit=person

while(1):

    print_table()

    while(1):

        if(turn==1):

            if cnt<2:

                r=random.randint(0, 2)

                c=random.randint(0, 2)

            else:

                i=rest[0]

                c_best=i%10-1

                r_best=i//10-1

                for i in rest:

                    h=0;rh=0;lh=0;uh=0;dh=0;d1=0;d2=0;d3=0;d4=0

                    c=i%10-1

                    r=i//10-1

                    if(c+1<=2):

                        rh+=hvalue(r,c+1)

                        if(c+2<=2):

                            rh+=hvalue(r,c+2)

                    if(c-1>=0):

                        lh+=hvalue(r,c-1)

                        if(c-2>=0):

                            lh+=hvalue(r,c-2)

                    if(r+1<=2):

                        dh+=hvalue(r+1,c)

                        if(r+2<=2):

                            dh+=hvalue(r+2,c)

                    if(r-1>=0):

                        uh+=hvalue(r-1,c)

                        if(r-2>=0):

                            uh+=hvalue(r-2,c)

                    if((r==0 and c==0) or (r==0 and c==2) or (r==2 and c==2) or (r==2 and c==0) or (r==1 and c==1)):

                        if((r-1>=0) and (c-1>=0)):

                            d1=hvalue(r-1,c-1)

                            if((r-2>=0) and (c-2>=0)):

                                d1+=hvalue(r-2,c-2)

                        if((r-1>=0) and (c+1<=2)):

                            d2=hvalue(r-1,c+1)

                            if((r-2>=0) and (c+2<=2)):

                                d2+=hvalue(r-2,c+2)

                        if((r+1<=2) and (c+1<=2)):

                            d3=hvalue(r+1,c+1)

                            if((r+2<=2) and (c+2<=2)):

                                d3+=hvalue(r+2,c+2)

                        if((r+1<=2) and (c-1>=0)):

                            d4=hvalue(r+1,c-1)

                            if((r+2<=2) and (c-2>=0)):

                                d4+=hvalue(r+2,c-2)

                    if ((lh+rh==6) or (uh+dh==6) or (d1+d3==6) or (d2+d4==6)):

                        best=h;r_best=r;c_best=c;break

                    if ((lh+rh==2) or (uh+dh==2) or (d1+d3==2) or (d2+d4==2)):

                        best=h;r_best=r;c_best=c

                r=r_best;c=c_best

            r+=1;c+=1

            value=pc

        else:

            print("Enter row ")

            r=int(input())

            print("Enter colum ")

            c=int(input());value=person

        p=(r)*10+(c)

        if(p not in done):

                done.append(p)

                table[r-1][c-1]=value

                break

    turn = 0 if turn==1 else 1

    rest.remove(p)

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

            print_table();break