import sys









#----------------------------------------------------------

# 해당 좌표에 들어갈 수 있는 숫자 list 반환해주는 함수

#----------------------------------------------------------

def make_candidate(pos):

  # 0~9까지 Flase/True로 구성된 list를 생성하고,

  # 사용하지 않는 0은 False, 나머지는 True로 초기화

  nums = [False] + [True for _ in range(9)]

  

  

  # 1. block 탐색

  x = (pos[0] // 3) * 3  # block 탐색 시작 row

  y = (pos[1] // 3) * 3  # block 탐색 시작 col

  

  # 해당 좌표가 포함된 block에서

  for i in range(x, x+3):

    for j in range(y, y+3):

      # 어떤 숫자가 있으면, 그 숫자에 해당하는 nums를 False로 바꿔준다.

      if puzzle[i][j]:

        nums[puzzle[i][j]] = False

        

  # 2. row, col 탐색

  for i in range(9):

    if puzzle[pos[0]][i]:

      nums[puzzle[pos[0]][i]] = False

    if puzzle[i][pos[1]]:

      nums[puzzle[i][pos[1]]] = False

      

  # 3. True로 남아있는 애들만 해당 번호 list로 만들어서 반환해주기

  return [i for i, e in enumerate(nums) if e]

#----------------------------------------------------------

# 해당 좌표에 숫자 다 넣어서 다음 좌표로 넘어가는 함수

#----------------------------------------------------------

def backtracking(k):

  global state  # 밖에서 선언해준 global 변수 state를 함께 사용한다고 선언

  

  # 더이상 공백이 없는 경우 (마지막)

  if k == len(zero_position):

    # 지금까지 완성된 스도쿠 출력

    for e in puzzle:

      print(''.join(list(map(str,e))))

    # 완성 여부 True로 변경

    state = True

    

  # 현재 공백 위치에서 실행될 부분

  else:

    # 현재 칸에 들어갈 수 있는 숫자들 알려줘! --> 각 숫자를 for문으로 돌리기

    for num in make_candidate(zero_position[k]):

      # 해당 좌표에 후보로 들어온 숫자 넣기

      puzzle[zero_position[k][0]][zero_position[k][1]] = num

      # 그 상태로 다음 공백으로 넘어가기

      backtracking(k+1)

      

      # 다음 공백탐색 함수가 끝나면 지금까지 완성됐는지 확인하기

      if state:

        break  # 완성됐으면 for문 끝내기 --> 함수도 끝나지

        

      # 완성이 안됐으면 지금 여기는 다시 공백으로 만들어주기

      puzzle[zero_position[k][0]][zero_position[k][1]] = 0
#----------------------------------------------------------

# 프로그램 실행

#----------------------------------------------------------

puzzle = []         # 퍼즐 list

zero_position = []  # 비어있는 칸의 좌표를 저장하는 list

state = False       # 스도쿠 완성 여부



# 스도쿠 문제 입력받기



# 0~8열까지 한줄씩 입력받기

for i in range(9):

  # 연속된 숫자 한 줄(문자열) 입력받아서 한자리씩 정수형으로 변형시켜서 row 배열 생성

  row = list(map(int, list(input())))

  

  # 그 배열을 puzzle list 뒤에 추가

  puzzle.append(row)

  

  # 공백인(==0) 칸이 발견되면 해당 좌표 [row, col]를 zero_position list에 추가해주기

  for j in range(9):

    if not row[j]:

      zero_position.append([i, j])



# 문제풀이 함수 실행

backtracking(0)



#한줄씩입력하기 

#103000509

#002109400

#000704000

#300502006

#060000050

#700803004

#000401000

#009205800

#804000107
