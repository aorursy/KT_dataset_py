import numpy as np

import math

import collections



'''

지게차 수량 

창고

aisle :  혼잡이 발생하는 곳을 복도 부분으로 한정, 지게차 작업시간 동안 점유되는 것으로 가정 

dock: dock이 비어있거나, 사용중인 truckNo 가 있는 order 만 지게차에 할당 가능 



monte carlo tree search

https://mongxmongx2.tistory.com/17



'''



#np.random.seed(1)



def distance(start_pos, target_pos):

    dist = math.fabs(start_pos.x - target_pos.x)

    dist += math.fabs(start_pos.y - target_pos.y)



    return dist





"""Pos 창고내 좌표를 표시하기 위함 

"""





class Pos:

    def __init__(self, x, y, z):

        self.x = x

        self.y = y

        self.z = z  # 선반의 층수 (작업시간 계산 시 활용)



    def __str__(self):

        return '(' + str(self.x) + ',' + str(self.y) + ',' + str(self.z) + ')'





"""Order  처리해야 하는 주문

 turck_no : 해당 주문은 동일 truck_no가 있는 Dock 으로 이동해야 함  

"""





class Order:

    def __init__(self, no, pos, aisle, truck_no):

        self.no = no

        self.pos = pos

        self.aisle = aisle

        self.truck_no = truck_no

        self.done = False

        self.finish_time = -1



    def __str__(self):

        return str(self.done) + ':Order ' + str(self.no) + ':pos=' + str(self.pos) + ':aisle=' + str(

            self.aisle) + ':Truck=' + str(self.truck_no)





class Forklift:

    def __init__(self, no, time_clock, pos):

        self.no = no

        self.time_clock = time_clock

        self.pos = pos

        self.order = None

        self.dest_dock = None   # Dock 번호 사용

        self.work_time = 0      #순수 작업 시간



        self.to_aisle_time = 0   #aisle 까지 가는 시간

        self.aisle_to_dock_time = 0  #aisle에서 dock까지 가는 시간



        self.move_in_aisle_time = 0   #aisle 입구에서 들어가는 시간

        self.picking_time = 0         #선반에서 상품 꺼내는 시간

        self.move_out_aisle_time = 0  #aisle 입구로 나오는 시간



        self.start_time_clock = 0

        self.finish_time_clock = 0    #지게차의 작업 종료시간

        self.release_aisle_time_clock = 0   #지게차가 공유자원(Aisle)을 해제하는 시간





    def __str__(self):

        return 'Forklift=' + str(self.no) + ':pos=' + str(self.pos) + ':dest_dock' + str(self.dest_dock)  + ':order=' + str(self.order.no)



    def allocate_order(self, order, dest_dock):

        # order 할당 시, 종료시간 계산



        self.order = order

        self.dest_dock = dest_dock  # 목적지 지정



        self.start_time_clock = self.time_clock.time_clock



        move_in_time = distance(self.pos, self.order.aisle.pos)  # aisle 입구까지만



        picking_in_out_time = (distance(self.order.aisle.pos, self.order.pos) ) * 2



        picking_time = self.order.pos.z * 5  # TODO   각 층별로 시간을 5씩 추가소요 가정, 실제 상황에 맞게 수정 필요

        move_out_time = distance(self.order.aisle.pos, self.dest_dock.pos)  # aisle입구에서 Dock 위치까지



        self.work_time = move_in_time + picking_in_out_time + picking_time + move_out_time



        aisle_arrival_time_clock = self.time_clock.time_clock + move_in_time



        base_time_clock = order.aisle.get_available_time_clock(aisle_arrival_time_clock)  # aisle쪽에 가용하게 되는 시간 확인



        self.release_aisle_time_clock = base_time_clock + picking_in_out_time + picking_time

        self.finish_time_clock = self.release_aisle_time_clock + move_out_time   # 최종 종료시간 설정됨

        self.order.aisle.append(self)  # 해당 Aisle 에 대기 추가



        print('forklift ', self.no, ': order=', self.order.no, ',finish_time_clock=', self.finish_time_clock, ',release_aisle_time_clock=', self.release_aisle_time_clock,  ',start_time_clock=', self.start_time_clock,

              ',move_in_time=', move_in_time , ',picking_in_out_time=', picking_in_out_time , ',picking_time=', picking_time , ',move_out_time=', move_out_time)





    def get_finish_time(self):

        return self.finish_time



    def set_clear(self):

        self.order = None

        self.dest_dock = None

        self.work_time = 0

        self.finish_time = 0





class Dock:

    def __init__(self, no, pos):

        self.no = no

        self.truck_no = -1

        self.pos = pos



    def __str__(self):

        return 'Dock=' + str(self.no) + ':pos=' + str(pos) + ':truck_no=' + str(self.truck_no)





class SimulationTimeClock:

    def __init__(self):

        self.time_clock = 0



    def __str__(self):

        return 'TimeClock=' + str(self.time_clock)



    def increase(self):

        self.time_clock += 1



    def get_time_clock(self):

        return self.time_clock





class Aisle:

    def __init__(self, no, time_clock, start_pos, length=5):

        self.no = no

        self.time_clock = time_clock

        self.pos = start_pos

        self.queue = collections.deque()

        self.length = length



    def __str__(self):

        return 'Aisle=' + str(self.no) + ':' + str(self.pos)



    def get_available_time_clock(self, arrival_time_clock):

        size = len(self.queue)

        available_time_clock = arrival_time_clock



        if size > 0:  # 대기 지게차가

            last_forklift = self.queue[size - 1]  # 마지막 지게차

            if arrival_time_clock < last_forklift.release_aisle_time_clock: #더 일찍 도착한 상황이면, 먼저 있는 지게차 시간부터 작업 가능

                available_time_clock = last_forklift.release_aisle_time_clock



        return available_time_clock



    def append(self, forklift):

        self.queue.append(forklift)



    def pop(self):

        forklift = self.queue.popleft()

        return forklift



    def remove_waiting_forklift(self, time_clock):



        while len(self.queue) > 0:

            forklift = self.queue[0]

            if forklift.release_aisle_time_clock <= time_clock :

                self.pop()

                print('Aisel=', self.no, ':Leave aisle time_clock=', time_clock, ',forklift=', forklift.no )

            else:

                break





class State:

    def __init__(self, forklifts, orders):

        # 현재 상태를 어떻게 표한할지 고민 중....

        self.orders = orders

        self.forklifts = forklifts

        self.time_hold = False



    def __str__(self):

        return 'State: orders=' + str(len(self.orders))





class Action:

    def __init__(self):

        self.forklift = None

        self.order = None



    def __init__(self, forklift, order):

        self.forklift = forklift

        self.order = order



    def __str__(self):

        return 'Action: forklift=' + str(super.forklift) + ':order=' + str(super.order)





class Warehouse:

    def __init__(self, time_clock, size, docks, aisles, forklifts):

        self.time_clock = time_clock

        self.size = size  # (x,y)

        self.orders = []  # 처리해야할 주문들

        self.available_orders = []

        self.docks = docks  # dock의 위치들

        self.aisles = aisles

        self.forklifts = forklifts

        self.finish_time_clock = -1



    def set_orders(self, orders):

        self.orders = orders

        self.available_orders = orders.copy()



    def remove_available_order(self, order):

        print('- pop order from available order:', order)

        self.available_orders.remove(order)



    def get_dest_dock(self, truck_no):

        dest_dock = None

        # 동일 truck_no가 배정된 dock 찾기

        for dock in self.docks:

            if dock.truck_no == truck_no:

                dest_dock = dock

                break

            if dock.truck_no < 0:

                dest_dock = dock



        #print('dest dock:', dest_dock)

        return dest_dock



    def increase_time(self):

        time_hold_flag = False

        #forklift 여분 확인, order 여유확인

        for f in self.forklifts:

            if f.order == None and len(self.available_orders) > 0:

                time_hold_flag = True



        if not time_hold_flag :

            self.time_clock.increase()  # time 증가





    def is_done(self):

        done = False

        # 남은 주문이 없고 # 지게차의 마지막 시간이 동일할때

        if len(self.available_orders) == 0 and self.finish_time_clock >= self.time_clock.time_clock:

            done = True



        return done





    def step(self, action):



        reward = -1

        done = False

        info = []



        #aisle 에서 완료된 forklift는 제거

        for aisle in self.aisles:

            aisle.remove_waiting_forklift(self.time_clock.time_clock)





        # forklifts 모두 체크하고, 종료되는 작업도 확인

        for f in self.forklifts:

            # forklift의 종료시간이 현재 시간 이하이면 종료처리

            if f.order != None and f.finish_time_clock <= self.time_clock.time_clock:

                print('* Complete work: Forklift',f,':order=',f.order.no)

                f.order.done = True  # 작업완료 표시

                f.order = None  # 작업할당 해제

                f.dest_dock = None



        forklift = action.forklift

        order = action.order



        if forklift != None and order != None:

            # dest_dock 찾기 (truck_no가 동일하거나, 비어있는 dock을 할당가능, dock은 지게차가 놓은뒤... 일정시간 이후 해제가능 )

            # 트럭배정과 해제는 나중에 더 고려해야 함

            dest_dock = self.get_dest_dock(order.truck_no)  #TODO dock에 트럭을 할당하는 기준 필요 (주문을 truck 별로 모아서 배정하는 것 같은 전략)



            dest_dock.truck_no = -1 #TODO 일단 바로 트럭은 해제 (나중에 독 배정로직 보완필요함)



            if dest_dock != None:

                self.remove_available_order(order)  # 가용 order에서 제거

                forklift.dest_dock = dest_dock

                forklift.pos = dest_dock.pos

                forklift.allocate_order(order, dest_dock)

                print('+ order allocation:f=', forklift, ':order=', order.no , ':dest_dock=', dest_dock.no )

                print('-' * 40 )



                # 창고의 종료시간을 최종 지게차 종료시간 기준으로 설정 (모든 가용주문이 처리된 이후 확인함)

                if forklift.finish_time_clock > self.finish_time_clock:

                    self.finish_time_clock = forklift.finish_time_clock



        # 상태: 남아있는 order, forklift, dock 상태

        state = State(self.forklifts, self.available_orders)



        # reawrd: 시간을 cost로 간주하고 order를 처리하는 총 시간을 줄이는 것을 목표로 함 (혼잡발생시, 대기시간으로 시간 증가됨)

        reward = -1



        self.increase_time()

        done = self.is_done()





        return (state, reward, done, info)





def get_action(warehouse):

    orders = warehouse.available_orders



    forklift = None

    order = None



    if len(orders) > 0:

        for f in warehouse.forklifts:

            if f.order == None:  # 가용한 지게차가 있는 경우

                forklift = f

                order = orders[0]

                #order = np.random.choice(orders)

                break





    return Action(forklift, order)





########################################################################################################################





time_clock = SimulationTimeClock() # 해당 창고내에서는 공유해야할 Time



size = (10,10) #창고 사이즈 (Grid 구조가정)



docks = [Dock(0, Pos(0,0,0)),

         Dock(1,Pos(9,0,0))  ]



aisles = [Aisle(0, time_clock, Pos(0,4,0), 5),

          Aisle(1, time_clock, Pos(5,4,0), 5),

          Aisle(2, time_clock, Pos(9,4,0), 5)

          ]



forklifts = [Forklift(0,time_clock, Pos(9,0,0)),

             Forklift(1, time_clock, Pos(9,0,0)) ]



orders = []

order_no = 0



#order 생성

for i in range(len(aisles)):# i = aisle 번호

    for j in range(2):

        #no, pos, aisle_no, truck_no

        pos = Pos(aisles[i].pos.x, aisles[i].pos.y + 1 + j, 1) # x, y, z

        truck_no = np.random.randint(0,2)

        aisle_no = i # np.random.randint(0,len(aisles))

        order = Order(order_no, pos, aisles[aisle_no], truck_no) #i번 aisle에 truck_no 기준 order

        order_no += 1

        orders.append(order)

        print('> Order create:', order)



#time_clock, size, docks,  aisles, forklifts

env = Warehouse(time_clock, size, docks, aisles, forklifts)

env.set_orders(orders)





done = False

state = None



while True:

    action = get_action(env)

    next_state, reward, done, info = env.step(action)  #state 에 현재 forklifts 의 상태필

    state = next_state



    if done :

        break



print('[Result]')

print('Finish time=', env.finish_time_clock,':orders=',len(env.available_orders))

print('Simulation is finished.' )



#TODO 테스트 코드 추가 필요함 (소스 변경 시, 점검할 수 있는 코드)
1+1