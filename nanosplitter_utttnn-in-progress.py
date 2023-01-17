class SmallBoard:
    def __init__(self):
        self.board = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        
    def takeSpot(self, index, player):
        if player == 1:
            self.board[index] = 1
        else:
            self.board[index] = 2
    
    def getBoard(self, player):
        res=[]
        for i in self.board:
            if i == player:
                res.append(1)
            elif i == 0:
                res.append(0)
            else:
                res.append(-1)
        return res
    
    def checkSpot(self, spot):
        return self.board[spot] == 0
    
    def checkWin(self, player):
        self.top = [self.board[0], self.board[1], self.board[2]]
        self.horzmid = [self.board[3], self.board[4], self.board[5]]
        self.bottom = [self.board[6], self.board[7], self.board[8]]
        self.left = [self.board[0], self.board[3], self.board[6]]
        self.vertmid = [self.board[1], self.board[4], self.board[7]]
        self.right = [self.board[2], self.board[5], self.board[8]]
        self.leftdiag = [self.board[0], self.board[4], self.board[8]]
        self.rightdiag = [self.board[2], self.board[4], self.board[6]]
        
        self.all = [self.top, self.horzmid, self.bottom, self.left, self.vertmid, self.right, self.leftdiag, self.rightdiag]
        self.emptycount = 0
        for i in self.all:
            self.emptycount += i.count(0)
            if i.count(player) == 3:
                return True
        
        if self.emptycount == 0:
            return False
    
    def getEmptySpaces(self):
        res = []
        if not (self.checkWin(1) or self.checkWin(2) or self.checkTie()):
            for i in range(len(self.board)):
                if self.board[i] == 0:
                    res.append(i)
        return res
    
    def checkValidBoard(self):
        return not (self.checkWin(1) or self.checkWin(2) or self.checkTie())
    
    def checkTie(self):
        if self.board.count(0) == 0:
            return True
        else:
            return False
class Move:
    def __init__(self, b, s):
        self.bindex = b
        self.sindex = s
    def __repr__(self):
        return "[" + str(self.bindex) + ", " + str(self.sindex) + "]"
class BigBoard:
    def __init__(self):
        self.board = []
        for i in range(9):
            self.board.append(SmallBoard())
    
    def makeMove(self, bindex, sindex, player):
        if (self.board[bindex].checkSpot(sindex)):
            self.board[bindex].takeSpot(sindex, player)
            return True
        else:
            return False
        
    def getValidMoves(self, sentTo):
        #print(sentTo)
        res = []
        if not self.board[sentTo].checkValidBoard() or sentTo == -1:
            for i in range(9):
                for m in self.board[i].getEmptySpaces():
                    res.append(Move(i, m))
        else:
            for i in self.board[sentTo].getEmptySpaces():
                res.append(Move(sentTo, i))
        return res
    
    def getBoardForPlayer(self, player):
        res = []
        for i in range(9):
            for s in self.board[i].getBoard(player):
                res.append(s)
        return res
    
    def getBoardForConsole(self):
        boards = []
        for i in range(9):
            boards.append(self.board[i].getBoard(1))
        
        for i in boards:
            for s in range(len(i)):
                if i[s] == -1:
                    i[s] = "X"
                elif i[s] == 1:
                    i[s] = "O"
                else:
                    i[s] = " "
        res = ""
        res += boards[0][0] + boards[0][1] + boards[0][2] + " | " + boards[1][0] + boards[1][1] + boards[1][2] + " | " + boards[2][0] + boards[2][1] + boards[2][2] + "\n"
        res += boards[0][3] + boards[0][4] + boards[0][5] + " | " + boards[1][3] + boards[1][4] + boards[1][5] + " | " + boards[2][3] + boards[2][4] + boards[2][5] + "\n"
        res += boards[0][6] + boards[0][7] + boards[0][8] + " | " + boards[1][6] + boards[1][7] + boards[1][8] + " | " + boards[2][6] + boards[2][7] + boards[2][8] + "\n"
        res += "---------------\n"
        res += boards[3][0] + boards[3][1] + boards[3][2] + " | " + boards[4][0] + boards[4][1] + boards[4][2] + " | " + boards[5][0] + boards[5][1] + boards[5][2] + "\n"
        res += boards[3][3] + boards[3][4] + boards[3][5] + " | " + boards[4][3] + boards[4][4] + boards[4][5] + " | " + boards[5][3] + boards[5][4] + boards[5][5] + "\n"
        res += boards[3][6] + boards[3][7] + boards[3][8] + " | " + boards[4][6] + boards[4][7] + boards[4][8] + " | " + boards[5][6] + boards[5][7] + boards[5][8] + "\n"
        res += "---------------\n"
        res += boards[6][0] + boards[6][1] + boards[6][2] + " | " + boards[7][0] + boards[7][1] + boards[7][2] + " | " + boards[8][0] + boards[8][1] + boards[8][2] + "\n"
        res += boards[6][3] + boards[6][4] + boards[6][5] + " | " + boards[7][3] + boards[7][4] + boards[7][5] + " | " + boards[8][3] + boards[8][4] + boards[8][5] + "\n"
        res += boards[6][6] + boards[6][7] + boards[6][8] + " | " + boards[7][6] + boards[7][7] + boards[7][8] + " | " + boards[8][6] + boards[8][7] + boards[8][8] + "\n"
        
        return res
    def checkWin(self):
        self.top = [self.board[0], self.board[1], self.board[2]]
        self.horzmid = [self.board[3], self.board[4], self.board[5]]
        self.bottom = [self.board[6], self.board[7], self.board[8]]
        self.left = [self.board[0], self.board[3], self.board[6]]
        self.vertmid = [self.board[1], self.board[4], self.board[7]]
        self.right = [self.board[2], self.board[5], self.board[8]]
        self.leftdiag = [self.board[0], self.board[4], self.board[8]]
        self.rightdiag = [self.board[2], self.board[4], self.board[6]]
        
        self.all = [self.top, self.horzmid, self.bottom, self.left, self.vertmid, self.right, self.leftdiag, self.rightdiag]
        self.emptycount = 0
        self.tiecount = 0
        self.wintotal1 = 0
        self.wintotal2 = 0
        self.winTotal = 0
        for i in self.all:
            self.wincount1 = 0
            self.wincount2 = 0
            for b in i:
                if b.checkWin(1):
                    self.wincount1 += 1
                if b.checkWin(2):
                    self.wincount2 += 1
                if b.checkTie():
                    self.tiecount += 1
            if self.wincount1 == 3:
                return 1
            if self.wincount2 == 3:
                return 2
            
        
        if self.tiecount + self.wintotal1 + self.wintotal2 == 9:
            if self.wintotal1 > self.wintotal2:
                return 1
            elif self.wintotal2 > self.wintotal1:
                return 2
            else:
                return 0    
        return -1
import random

class Player:
    def makeMove(self, board, validMoves):
#         print("Valid Moves:", validMoves)
        return random.choice(validMoves)
class GameResult:
    def __init__(self, winner, turns):
        self.winner = winner
        self.turns = turns
        if self.winner == 1:
            self.score1 = 90 - self.turns
            self.score2 = 0
            self.winner = "O"
        if self.winner == 2:
            self.score2 = 90 - self.turns
            self.score1 = 0
            self.winner = "X"
        if self.winner == 0:
            self.score1 = 1
            self.score2 = 1
            self.winner = "Tie"
    def __repr__(self):
        return "Winner: " + str(self.winner) + " Turns: " + str(self.turns) + " Score1: " + str(self.score1) + " Score2: " + str(self.score2)
class Game:
    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2
        self.board = BigBoard()
        
    def playGame(self):
        self.turn = 0
        while True:
#             try:
            if (self.turn == 0):
                self.move1 = self.player1.makeMove(self.board.getBoardForPlayer(1), self.board.getValidMoves(-1))
                self.board.makeMove(self.move1.bindex, self.move1.sindex, 1)
#                 print("O Move made:", self.move1)
#                 print(self.board.getBoardForConsole())
            elif (self.turn % 2 == 0):
                self.move1 = self.player1.makeMove(self.board.getBoardForPlayer(1), self.board.getValidMoves(self.move2.sindex))
                self.board.makeMove(self.move1.bindex, self.move1.sindex, 1)
#                 print("O Move made:", self.move1)
#                 print(self.board.getBoardForConsole())
            else:
                self.move2 = self.player2.makeMove(self.board.getBoardForPlayer(2), self.board.getValidMoves(self.move1.sindex))
                self.board.makeMove(self.move2.bindex, self.move2.sindex, 2)
#                 print("X Move made:", self.move2)
#                 print(self.board.getBoardForConsole())
                
            self.turn += 1
            self.winMessage = self.board.checkWin()
            #print(self.winMessage)
            if self.winMessage == 1:
                self.gameres = GameResult(1, self.turn)
                print(self.board.getBoardForConsole())
                return self.gameres
            elif self.winMessage == 2:
                self.gameres = GameResult(2, self.turn)
                print(self.board.getBoardForConsole())
                return self.gameres
            elif  self.board.getValidMoves(-1) == []:
                self.gameres = GameResult(0, self.turn)
                print(self.board.getBoardForConsole())
                return self.gameres
#             except:
#                 print(self.board.getBoardForConsole())
#                 return
        
        
game = Game(Player(), Player())

res = game.playGame()

print(res)


